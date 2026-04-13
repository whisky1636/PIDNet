import sys
import os
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QFrame, QProgressBar, QMessageBox, QSlider, QStyle, QStyleOptionSlider, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QDialog)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
import cv2
import numpy as np
from predictor import RoadDamagePredictor
from db_manager import DBManager

class ClickableSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.setSliderPosition(self.minimum() + ((self.maximum() - self.minimum()) * event.x()) // self.width())
            self.sliderMoved.emit(self.value())
            event.accept()
        super().mousePressEvent(event)

class VideoPlayer(QThread):
    frame_ready = pyqtSignal(QImage, object) # 第二个参数可以是 QImage 或 None
    finished = pyqtSignal()
    position_changed = pyqtSignal(int)
    fps_ready = pyqtSignal(float)

    def __init__(self, original_path, result_path=None):
        super().__init__()
        self.original_path = original_path
        self.result_path = result_path
        self.running = True
        self.paused = False
        self.seek_pos = -1

    def run(self):
        cap_orig = cv2.VideoCapture(self.original_path)
        cap_res = cv2.VideoCapture(self.result_path) if self.result_path else None
        
        total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_orig.get(cv2.CAP_PROP_FPS)
        self.fps_ready.emit(fps)
        
        while self.running:
            if self.paused:
                self.msleep(100)
                continue

            if self.seek_pos >= 0:
                cap_orig.set(cv2.CAP_PROP_POS_FRAMES, self.seek_pos)
                if cap_res:
                    cap_res.set(cv2.CAP_PROP_POS_FRAMES, self.seek_pos)
                self.seek_pos = -1

            curr_frame_idx = int(cap_orig.get(cv2.CAP_PROP_POS_FRAMES))
            ret_o, frame_o = cap_orig.read()
            ret_r, frame_r = (cap_res.read() if cap_res else (False, None))
            
            if not ret_o:
                break
                
            # 转换为 QImage 原始
            rgb_o = cv2.cvtColor(frame_o, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_o.shape
            q_img_o = QImage(rgb_o.data, w, h, ch * w, QImage.Format_RGB888)
            
            # 转换为 QImage 结果 (如果存在)
            q_img_r = None
            if ret_r:
                rgb_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
                h2, w2, ch2 = rgb_r.shape
                q_img_r = QImage(rgb_r.data, w2, h2, ch2 * w2, QImage.Format_RGB888)
            
            self.frame_ready.emit(q_img_o.copy(), q_img_r.copy() if q_img_r else None)
            self.position_changed.emit(curr_frame_idx)
            self.msleep(33)
            
        cap_orig.release()
        if cap_res: cap_res.release()
        self.finished.emit()

    def stop(self):
        self.running = False

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def seek(self, frame_idx):
        self.seek_pos = frame_idx

class SegmentationThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int) # 当前已完成数, 总数
    
    def __init__(self, predictor, path, mode='image', save_dir=None):
        super().__init__()
        self.predictor = predictor
        self.path = path
        self.mode = mode # 'image', 'folder', 'video'
        self.save_dir = save_dir

    def run(self):
        try:
            if self.mode == 'folder':
                def progress_cb(current, total):
                    self.progress.emit(current, total)
                
                target_dir, count = self.predictor.predict_folder(
                    self.path, self.save_dir, progress_callback=progress_cb
                )
                self.finished.emit(('folder', target_dir, count))
            elif self.mode == 'video':
                def progress_cb(current, total):
                    self.progress.emit(current, total)
                
                target_path, count = self.predictor.predict_video(
                    self.path, self.save_dir, progress_callback=progress_cb
                )
                self.finished.emit(('video', target_path, count))
            else:
                prediction = self.predictor.predict(self.path)
                self.finished.emit(('image', prediction))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class MainWindow(QWidget):
    def __init__(self, username="Guest"):
        super().__init__()
        self.username = username
        self.CONFIG_FILE = "app_config.json"
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._drag_pos = None
        
        self.img_path = None
        self.folder_path = None
        self.video_path = None
        self.result_video_path = None
        self.video_player_thread = None
        self.process_mode = 'image' # 'image', 'folder', 'video'
        self.video_fps = 30.0
        
        # 文件夹模式相关属性
        self.folder_images = []
        self.current_folder_idx = -1
        self.folder_result_dir = None
        
        # 加载保存的目录或使用默认值
        self.load_config()
        
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except:
                self.save_dir = os.path.join(os.getcwd(), 'output')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            
        # 初始化预测模型
        try:
            self.predictor = RoadDamagePredictor(model_path='models/best.pt')
        except Exception as e:
            print(self, "错误", f"无法加载模型: {e}")
            self.predictor = None
            
        self.init_ui()

    def load_config(self):
        """加载应用程序配置"""
        default_save_dir = os.path.join(os.getcwd(), 'output')
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.save_dir = config.get("save_dir", default_save_dir)
                    return
            except Exception as e:
                print(f"加载配置失败: {e}")
        self.save_dir = default_save_dir

    def save_config(self):
        """保存应用程序配置"""
        config = {"save_dir": self.save_dir}
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存配置失败: {e}")

    def init_ui(self):
        self.setFixedSize(1100, 750)
        
        # 外层圆角容器
        self.container = QFrame(self)
        self.container.setObjectName("MainForm")
        self.container.setGeometry(10, 10, 1080, 730)
        self.container.setStyleSheet("#MainForm { background-color: #f7f9fc; border-radius: 20px; border: 1px solid #ddd; }")
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 自定义标题栏
        title_bar = QWidget()
        title_bar.setFixedHeight(50)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 0, 10, 0)
        
        title_label = QLabel("道路病害分割识别系统 - 处理终端")
        title_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 16px;")
        
        min_btn = QPushButton("—")
        min_btn.setFixedSize(30, 30)
        min_btn.setStyleSheet("QPushButton { border:none; color: #555; font-size: 14px; } QPushButton:hover { background-color: #eee; border-radius: 5px; }")
        min_btn.clicked.connect(self.showMinimized)
        
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet("QPushButton { border:none; color: #555; font-size: 14px; } QPushButton:hover { background-color: #e74c3c; color: white; border-radius: 5px; }")
        close_btn.clicked.connect(self.close)
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(min_btn)
        title_layout.addWidget(close_btn)
        layout.addWidget(title_bar)

        # 内容区
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(20, 10, 20, 20)
        content_layout.setSpacing(20)

        # 左侧面板：控制按钮
        left_panel = QFrame()
        left_panel.setFixedWidth(240)
        left_panel.setStyleSheet("QFrame { background: white; border-radius: 12px; border: 1px solid #e1e7ef; }")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 20, 15, 20)
        left_layout.setSpacing(15)

        btn_style = """
            QPushButton {
                background-color: #3498db; color: white; border-radius: 6px; 
                padding: 12px; font-weight: bold; border: none; font-size: 13px;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """
        
        self.upload_btn = QPushButton("📁 上传病害图像")
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.setStyleSheet(btn_style)
        self.upload_btn.clicked.connect(self.upload_image)
        
        self.upload_folder_btn = QPushButton("📂 上传病害文件夹")
        self.upload_folder_btn.setCursor(Qt.PointingHandCursor)
        self.upload_folder_btn.setStyleSheet(btn_style.replace("#3498db", "#e67e22").replace("#2980b9", "#d35400"))
        self.upload_folder_btn.clicked.connect(self.upload_folder)

        self.upload_video_btn = QPushButton("🎥 上传病害视频")
        self.upload_video_btn.setCursor(Qt.PointingHandCursor)
        self.upload_video_btn.setStyleSheet(btn_style.replace("#3498db", "#1abc9c").replace("#2980b9", "#16a085"))
        self.upload_video_btn.clicked.connect(self.upload_video)
        
        self.set_save_btn = QPushButton("⚙️ 设置保存目录")
        self.set_save_btn.setStyleSheet(btn_style.replace("#3498db", "#9b59b6").replace("#2980b9", "#8e44ad"))
        self.set_save_btn.clicked.connect(self.set_save_directory)

        self.start_btn = QPushButton("🚀 开始智能分割")
        self.start_btn.setStyleSheet(btn_style.replace("#3498db", "#2ecc71").replace("#2980b9", "#27ae60"))
        self.start_btn.clicked.connect(self.start_segmentation)
        self.start_btn.setEnabled(False)

        self.history_btn = QPushButton("📊 查看分割记录")
        self.history_btn.setStyleSheet(btn_style.replace("#3498db", "#f39c12").replace("#2980b9", "#e67e22"))
        self.history_btn.clicked.connect(self.show_history)

        info_label = QLabel("保存路径:")
        info_label.setStyleSheet("border:none; color: #7f8c8d; font-size: 12px; margin-top: 10px;")
        self.path_display = QLabel(self.save_dir)
        self.path_display.setWordWrap(True)
        self.path_display.setStyleSheet("border:none; color: #2c3e50; font-size: 11px; font-style: italic;")

        left_layout.addWidget(self.upload_btn)
        left_layout.addWidget(self.upload_folder_btn)
        left_layout.addWidget(self.upload_video_btn)
        left_layout.addWidget(self.set_save_btn)
        left_layout.addWidget(self.history_btn)
        left_layout.addStretch()
        left_layout.addWidget(info_label)
        left_layout.addWidget(self.path_display)
        left_layout.addWidget(self.start_btn)
        
        # 右侧面板：图像展示
        right_panel = QFrame()
        right_panel.setStyleSheet("QFrame { background: #f0f2f5; border-radius: 12px; border: 1px solid #e1e7ef; }")
        right_layout = QVBoxLayout(right_panel)
        
        img_display_area = QHBoxLayout()
        self.orig_view = QLabel("待采集图像")
        self.res_view = QLabel("分割结果图")
        for view in [self.orig_view, self.res_view]:
            view.setAlignment(Qt.AlignCenter)
            view.setStyleSheet("background: #dcdde1; border-radius: 8px; color: #7f8c8d; border: 2px dashed #bdc3c7;")
            img_display_area.addWidget(view)
        
        # 视频控制条
        self.video_controls = QWidget()
        self.video_controls.setVisible(False)
        video_ctrl_layout = QHBoxLayout(self.video_controls)
        video_ctrl_layout.setContentsMargins(10, 0, 10, 5)
        
        self.play_pause_btn = QPushButton("▶") # 播放/暂停
        self.play_pause_btn.setFixedSize(40, 30)
        self.play_pause_btn.setStyleSheet("QPushButton { background-color: #34495e; color: white; border-radius: 4px; } QPushButton:hover { background-color: #2c3e50; }")
        self.play_pause_btn.clicked.connect(self.toggle_video_play)
        
        self.video_slider = ClickableSlider(Qt.Horizontal)
        self.video_slider.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #bbb; height: 6px; background: #ddd; border-radius: 3px; }
            QSlider::handle:horizontal { background: #3498db; border: 1px solid #3498db; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        self.video_slider.sliderMoved.connect(self.seek_video)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("border:none; color: #34495e; font-size: 11px; font-weight: bold; min-width: 80px;")
        
        video_ctrl_layout.addWidget(self.play_pause_btn)
        video_ctrl_layout.addWidget(self.video_slider)
        video_ctrl_layout.addWidget(self.time_label)

        # 文件夹控制条
        self.folder_controls = QWidget()
        self.folder_controls.setVisible(False)
        folder_ctrl_layout = QHBoxLayout(self.folder_controls)
        folder_ctrl_layout.setContentsMargins(10, 0, 10, 5)

        self.prev_btn = QPushButton("◀ 上一张")
        self.prev_btn.setFixedSize(80, 30)
        self.prev_btn.setStyleSheet("QPushButton { background-color: #34495e; color: white; border-radius: 4px; } QPushButton:hover { background-color: #2c3e50; }")
        self.prev_btn.clicked.connect(self.prev_image)

        self.next_btn = QPushButton("下一张 ▶")
        self.next_btn.setFixedSize(80, 30)
        self.next_btn.setStyleSheet("QPushButton { background-color: #34495e; color: white; border-radius: 4px; } QPushButton:hover { background-color: #2c3e50; }")
        self.next_btn.clicked.connect(self.next_image)

        self.folder_slider = QSlider(Qt.Horizontal)
        self.folder_slider.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #bbb; height: 6px; background: #ddd; border-radius: 3px; }
            QSlider::handle:horizontal { background: #e67e22; border: 1px solid #e67e22; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        self.folder_slider.valueChanged.connect(self.jump_to_image)

        self.folder_info_label = QLabel("0 / 0")
        self.folder_info_label.setStyleSheet("border:none; color: #34495e; font-size: 11px; font-weight: bold; min-width: 60px;")

        folder_ctrl_layout.addWidget(self.prev_btn)
        folder_ctrl_layout.addWidget(self.folder_slider)
        folder_ctrl_layout.addWidget(self.next_btn)
        folder_ctrl_layout.addWidget(self.folder_info_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { height: 8px; border-radius: 4px; text-align: center; } QProgressBar::chunk { background-color: #2ecc71; border-radius: 4px; }")

        right_layout.addLayout(img_display_area)
        right_layout.addWidget(self.video_controls)
        right_layout.addWidget(self.folder_controls)
        right_layout.addWidget(self.progress_bar)
        
        content_layout.addWidget(left_panel)
        content_layout.addWidget(right_panel)
        layout.addWidget(content)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if Qt.LeftButton and self._drag_pos:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择病害图像", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.img_path = file_path
            self.process_mode = 'image'
            self.folder_path = None
            self.video_path = None
            self.video_controls.setVisible(False)
            self.folder_controls.setVisible(False)
            if self.video_player_thread:
                self.video_player_thread.stop()
            
            pixmap = QPixmap(file_path)
            self.orig_view.setPixmap(pixmap.scaled(self.orig_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.orig_view.setStyleSheet("background: black; border-radius: 8px; border: none;")
            self.res_view.setPixmap(QPixmap())
            self.res_view.setText("等待分割...")
            self.start_btn.setEnabled(True)

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择病害图像文件夹", "")
        if folder_path:
            self.folder_path = folder_path
            self.process_mode = 'folder'
            self.img_path = None
            self.video_path = None
            self.video_controls.setVisible(False)
            self.folder_result_dir = None # 重置
            if self.video_player_thread:
                self.video_player_thread.stop()
            
            # 获取文件夹内的所有有效图片
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
            self.folder_images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
            self.folder_images.sort() # 排序保证浏览顺序
            
            if self.folder_images:
                self.current_folder_idx = 0
                self.update_folder_preview()
                self.folder_controls.setVisible(True)
                self.folder_slider.setRange(0, len(self.folder_images) - 1)
                self.folder_slider.setValue(0)
                self.folder_info_label.setText(f"1 / {len(self.folder_images)}")
            else:
                self.orig_view.setText("文件夹中无有效图片")
                self.current_folder_idx = -1
                self.folder_controls.setVisible(False)

            self.res_view.setPixmap(QPixmap())
            self.res_view.setText("等待批量分割...")
            self.start_btn.setEnabled(True if self.folder_images else False)

    def update_folder_preview(self):
        """显示文件夹中的当前图片及其对应的结果图（如果已生成）"""
        if self.current_folder_idx < 0 or not self.folder_images:
            return
            
        img_name = self.folder_images[self.current_folder_idx]
        img_path = os.path.join(self.folder_path, img_name)
        
        # 显示原始图
        pixmap_o = QPixmap(img_path)
        if not pixmap_o.isNull():
            self.orig_view.setPixmap(pixmap_o.scaled(self.orig_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.orig_view.setStyleSheet("background: black; border-radius: 8px; border: none;")
        
        # 显示结果图 (如果已经完成分割且结果存在)
        if self.folder_result_dir and os.path.exists(self.folder_result_dir):
            res_path = os.path.join(self.folder_result_dir, f"result_{img_name}")
            if os.path.exists(res_path):
                pixmap_r = QPixmap(res_path)
                self.res_view.setPixmap(pixmap_r.scaled(self.res_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.res_view.setStyleSheet("background: black; border-radius: 8px; border: none;")
            else:
                self.res_view.setPixmap(QPixmap())
                # 如果这个文件夹被处理过但没有这张图片的结果（例如中途停止），提示
                self.res_view.setText(f"未找到结果图:\nresult_{img_name}")
        else:
            self.res_view.setPixmap(QPixmap())
            self.res_view.setText("尚未批量处理\n处理后可在此查看结果")

    def prev_image(self):
        if self.current_folder_idx > 0:
            self.current_folder_idx -= 1
            self.folder_slider.setValue(self.current_folder_idx)
            self.update_folder_preview()
            self.folder_info_label.setText(f"{self.current_folder_idx + 1} / {len(self.folder_images)}")

    def next_image(self):
        if self.current_folder_idx < len(self.folder_images) - 1:
            self.current_folder_idx += 1
            self.folder_slider.setValue(self.current_folder_idx)
            self.update_folder_preview()
            self.folder_info_label.setText(f"{self.current_folder_idx + 1} / {len(self.folder_images)}")

    def jump_to_image(self, val):
        if 0 <= val < len(self.folder_images) and val != self.current_folder_idx:
            self.current_folder_idx = val
            self.update_folder_preview()
            self.folder_info_label.setText(f"{self.current_folder_idx + 1} / {len(self.folder_images)}")

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择病害视频", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            self.video_path = file_path
            self.result_video_path = None 
            self.process_mode = 'video'
            self.img_path = None
            self.folder_path = None
            self.video_controls.setVisible(True) # 视频控件此时必须可见
            self.folder_controls.setVisible(False)
            
            self.orig_view.setPixmap(QPixmap())
            self.orig_view.setText("加载视频中...")
            self.res_view.setPixmap(QPixmap())
            self.res_view.setText("等待分割...")
            self.start_btn.setEnabled(True)
            
            if self.video_player_thread and self.video_player_thread.isRunning():
                self.video_player_thread.stop()
                self.video_player_thread.wait()
            
            self.start_video_playback()

    def show_history(self):
        """显示历史记录窗口"""
        dialog = QDialog(self)
        dialog.setWindowTitle("分割历史记录")
        dialog.setMinimumSize(850, 500)
        layout = QVBoxLayout(dialog)
        
        table = QTableWidget()
        records = DBManager.get_user_records(self.username)
        table.setRowCount(len(records))
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["ID", "类型", "记录时间", "原始路径", "操作"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        for i, rec in enumerate(records):
            table.setItem(i, 0, QTableWidgetItem(str(rec['id'])))
            table.setItem(i, 1, QTableWidgetItem(rec['task_type']))
            table.setItem(i, 2, QTableWidgetItem(str(rec['created_at'])))
            table.setItem(i, 3, QTableWidgetItem(rec['original_path']))
            
            view_btn = QPushButton("查看对比数据")
            # 按钮闭包处理 record 参数
            view_btn.clicked.connect(lambda checked, r=rec: self.view_record_detail(r))
            table.setCellWidget(i, 4, view_btn)
            
        layout.addWidget(table)
        dialog.exec_()

    def view_record_detail(self, record):
        """查看单条记录的前后对比结果"""
        task_type = record['task_type']
        orig_path = record['original_path']
        res_path = record['result_path']
        
        if not os.path.exists(orig_path) or (task_type != 'folder' and not os.path.exists(res_path)):
            QMessageBox.warning(None, "文件损坏", "该路径下的原始文件或结果已不存在。")
            return
            
        detail_dialog = QDialog(self)
        detail_dialog.setWindowTitle(f"记录详情 - {task_type}")
        detail_dialog.setMinimumSize(1100, 700)
        d_layout = QVBoxLayout(detail_dialog)
        
        # 路径信息显示
        info_label = QLabel(f"<b>原始路径:</b> {orig_path}<br><b>结果路径:</b> {res_path}")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #2c3e50; font-size: 11px; padding: 5px;")
        d_layout.addWidget(info_label)
        
        content_box = QHBoxLayout()
        v_orig = QLabel("原始视图")
        v_res = QLabel("结果视图")
        for v in [v_orig, v_res]:
            v.setAlignment(Qt.AlignCenter)
            v.setStyleSheet("background: #000; border-radius: 8px; color: white; min-height: 400px;")
            content_box.addWidget(v)
        d_layout.addLayout(content_box)
        
        if task_type == 'image':
            p_orig = QPixmap(orig_path)
            p_res = QPixmap(res_path)
            v_orig.setPixmap(p_orig.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            v_res.setPixmap(p_res.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif task_type == 'video':
            # 视频同步播放逻辑
            ctrl_layout = QHBoxLayout()
            play_btn = QPushButton("播放")
            slider = ClickableSlider(Qt.Horizontal)
            time_lbl = QLabel("00:00 / 00:00")
            ctrl_layout.addWidget(play_btn)
            ctrl_layout.addWidget(slider)
            ctrl_layout.addWidget(time_lbl)
            d_layout.addLayout(ctrl_layout)
            
            # 使用现有的 VideoPlayer 类
            player = VideoPlayer(orig_path, res_path)
            
            def update_ui(q_img_o, q_img_r):
                # 检查 QLabel 是否仍然存在，防止窗口关闭后回调报错
                try:
                    if not v_orig.isVisible(): return
                    v_orig.setPixmap(QPixmap.fromImage(q_img_o).scaled(v_orig.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    if q_img_r:
                        v_res.setPixmap(QPixmap.fromImage(q_img_r).scaled(v_res.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                except RuntimeError:
                    pass

            def update_slider(pos):
                try:
                    if not slider.isVisible(): return
                    slider.setValue(pos)
                    # 避免在每一帧都进行 VideoCapture 操作，可以从 player 获取或缓存
                    # 这里暂且通过 try-except 保护
                    cap = cv2.VideoCapture(orig_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    curr_sec = int(pos / fps) if fps > 0 else 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    total_sec = int(total_frames / fps) if fps > 0 else 0
                    time_lbl.setText(f"{curr_sec//60:02d}:{curr_sec%60:02d} / {total_sec//60:02d}:{total_sec%60:02d}")
                    cap.release()
                except (RuntimeError, Exception):
                    pass

            player.frame_ready.connect(update_ui)
            player.position_changed.connect(update_slider)
            
            cap = cv2.VideoCapture(orig_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            slider.setMaximum(total)
            cap.release()
            
            play_btn.clicked.connect(lambda: play_btn.setText("暂停" if player.toggle_pause() else "播放"))
            slider.sliderMoved.connect(player.seek)
            
            # 窗口关闭时停止播放器
            detail_dialog.finished.connect(player.stop)
            player.start()
            
        elif task_type == 'folder':
            # 文件夹同步查看逻辑
            ctrl_layout = QHBoxLayout()
            prev_btn = QPushButton("◀ 上一张")
            next_btn = QPushButton("下一张 ▶")
            slider = ClickableSlider(Qt.Horizontal)
            info_lbl = QLabel("0 / 0")
            
            for btn in [prev_btn, next_btn]:
                btn.setFixedSize(80, 30)
                btn.setStyleSheet("background-color: #34495e; color: white; border-radius: 4px;")
            
            ctrl_layout.addWidget(prev_btn)
            ctrl_layout.addWidget(slider)
            ctrl_layout.addWidget(next_btn)
            ctrl_layout.addWidget(info_lbl)
            d_layout.addLayout(ctrl_layout)
            
            # 获取图片列表
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
            f_images = sorted([f for f in os.listdir(orig_path) if f.lower().endswith(valid_exts)])
            
            def update_folder_view(idx):
                if not 0 <= idx < len(f_images): return
                img_name = f_images[idx]
                
                # 原始图
                p_o = QPixmap(os.path.join(orig_path, img_name))
                v_orig.setPixmap(p_o.scaled(v_orig.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                # 结果图
                res_img_path = os.path.join(res_path, f"result_{img_name}")
                if os.path.exists(res_img_path):
                    p_r = QPixmap(res_img_path)
                    v_res.setPixmap(p_r.scaled(v_res.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                else:
                    v_res.setPixmap(QPixmap())
                    v_res.setText(f"未找到结果:\nresult_{img_name}")
                
                info_lbl.setText(f"{idx + 1} / {len(f_images)}")
                slider.setValue(idx)

            if f_images:
                slider.setRange(0, len(f_images) - 1)
                # 初始显示第一张
                # 使用 QTimer 确保布局完成后再加载图片以获得正确的 size()
                QTimer.singleShot(100, lambda: update_folder_view(0))
                
                prev_btn.clicked.connect(lambda: update_folder_view(slider.value() - 1))
                next_btn.clicked.connect(lambda: update_folder_view(slider.value() + 1))
                slider.valueChanged.connect(update_folder_view)
            else:
                v_orig.setText("文件夹中无有效图片")
            
        detail_dialog.exec_()

    def start_video_playback(self):
        """开始播放原始视频 (如果有结果则同步播放)"""
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            self.video_slider.setRange(0, total_frames - 1)
            self.video_slider.setValue(0)
            self.video_controls.setVisible(True)
            self.play_pause_btn.setText("⏸")
            
            self.video_player_thread = VideoPlayer(self.video_path, self.result_video_path)
            self.video_player_thread.frame_ready.connect(self.display_video_frames)
            self.video_player_thread.position_changed.connect(self.update_video_info)
            self.video_player_thread.fps_ready.connect(self.set_video_fps)
            self.video_player_thread.finished.connect(self.on_video_finished)
            self.video_player_thread.start()

    def set_video_fps(self, fps):
        self.video_fps = fps

    def update_video_info(self, current_frame):
        """更新进度条和时间标签"""
        self.video_slider.setValue(current_frame)
        total_frames = self.video_slider.maximum() + 1
        
        curr_sec = int(current_frame / self.video_fps)
        total_sec = int(total_frames / self.video_fps)
        
        curr_time = f"{curr_sec // 60:02d}:{curr_sec % 60:02d}"
        total_time = f"{total_sec // 60:02d}:{total_sec % 60:02d}"
        self.time_label.setText(f"{curr_time} / {total_time}")

    def toggle_video_play(self):
        if self.video_player_thread:
            is_paused = self.video_player_thread.toggle_pause()
            self.play_pause_btn.setText("▶" if is_paused else "⏸")

    def seek_video(self, pos):
        if self.video_player_thread:
            self.video_player_thread.seek(pos)

    def display_video_frames(self, q_img_o, q_img_r):
        """显示视频帧"""
        p_o = QPixmap.fromImage(q_img_o).scaled(self.orig_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.orig_view.setPixmap(p_o)
        self.orig_view.setStyleSheet("background: black; border-radius: 8px; border: none;")
        
        if q_img_r:
            p_r = QPixmap.fromImage(q_img_r).scaled(self.res_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.res_view.setPixmap(p_r)
            self.res_view.setStyleSheet("background: black; border-radius: 8px; border: none;")
        else:
            self.res_view.setPixmap(QPixmap())
            self.res_view.setText("尚未分割视频\n结果将在此显示")
            self.res_view.setStyleSheet("background: #dcdde1; border-radius: 8px; color: #7f8c8d; border: 2px dashed #bdc3c7;")

    def on_video_finished(self):
        """视频播放结束，重播"""
        if self.video_player_thread and self.video_player_thread.running:
            self.start_video_playback()

    def set_save_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录", self.save_dir)
        if dir_path:
            self.save_dir = dir_path
            self.path_display.setText(dir_path)
            self.save_config() # 保存新设置的目录到配置文件

    def start_segmentation(self):
        target_path = None
        if self.process_mode == 'image':
            target_path = self.img_path
        elif self.process_mode == 'folder':
            target_path = self.folder_path
        elif self.process_mode == 'video':
            target_path = self.video_path
            
        if not target_path or not self.predictor:
            return
            
        self.start_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.upload_folder_btn.setEnabled(False)
        self.upload_video_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        if self.process_mode == 'image':
            self.progress_bar.setRange(0, 0) # 忙碌状态
            self.thread = SegmentationThread(self.predictor, target_path, mode='image')
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.thread = SegmentationThread(self.predictor, target_path, mode=self.process_mode, save_dir=self.save_dir)
            self.thread.progress.connect(self.update_progress)
            
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.error.connect(self.on_processing_error)
        self.thread.start()

    def update_progress(self, current, total):
        if total > 0:
            val = int((current / total) * 100)
            self.progress_bar.setValue(val)
            prefix = "正在处理视频..." if self.process_mode == 'video' else "正在批量处理..."
            self.res_view.setText(f"{prefix}\n({current}/{total})")

    def on_processing_finished(self, result):
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.upload_folder_btn.setEnabled(True)
        self.upload_video_btn.setEnabled(True)
        
        res_type, *data = result
        
        if res_type == 'folder':
            target_dir, count = data
            self.folder_result_dir = target_dir # 关键：保存结果目录
            self.res_view.setText(f"批量处理完成!\n处理总数: {count}")
            # 更新预览，此时同步结果图
            self.update_folder_preview()
            
            # 保存到数据库
            DBManager.add_segmentation_record(self.username, 'folder', self.folder_path, target_dir)
            QMessageBox.information(self, "完成", f"批量识别成功！\n共处理 {count} 张图片。\n结果已保存至:\n{target_dir}\n您可以通过下方控件同步浏览结果。")
        elif res_type == 'video':
            target_path, count = data
            self.result_video_path = target_path
            self.res_view.setText(f"视频处理完成!\n处理总数: {count} 帧")
            # 保存到数据库
            DBManager.add_segmentation_record(self.username, 'video', self.video_path, target_path)
            QMessageBox.information(self, "完成", f"视频识别成功！\n共处理 {count} 帧。\n结果视频保存至:\n{target_path}\n点击上方视图开始同步播放。")
            
            # 如果有之前的播放线程，先停止
            if self.video_player_thread and self.video_player_thread.isRunning():
                self.video_player_thread.stop()
                self.video_player_thread.wait()
            
            # 开启视频预览播放
            self.start_video_playback()
        else:
            prediction = data[0]
            # 保存结果
            save_name = f"result_{os.path.basename(self.img_path)}"
            save_path = os.path.join(self.save_dir, save_name)
            self.predictor.save_result(prediction, save_path)
            
            # 保存到数据库
            DBManager.add_segmentation_record(self.username, 'image', self.img_path, save_path)
            
            # 显示结果
            pixmap = QPixmap(save_path)
            self.res_view.setPixmap(pixmap.scaled(self.res_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.res_view.setStyleSheet("background: black; border-radius: 8px; border: none;")
            
            QMessageBox.information(self, "完成", f"识别成功！结果已保存至:\n{save_path}")

    def closeEvent(self, event):
        """关闭窗口时清理线程"""
        if self.video_player_thread:
            self.video_player_thread.stop()
            self.video_player_thread.wait()
        event.accept()

    def on_processing_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.upload_folder_btn.setEnabled(True)
        self.upload_video_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{error_msg}")

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    from main_auth import AuthWindow
    app = QApplication(sys.argv)
    window = AuthWindow()
    window.show()
    sys.exit(app.exec_())
