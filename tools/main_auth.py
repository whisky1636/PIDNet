import sys
import json
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLineEdit, QPushButton, QLabel, QMessageBox, QFrame, QCheckBox)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QBrush, QIcon
from db_manager import DBManager
from main_window import MainWindow

# 通用样式设计
STYLING = """
    * {
        font-family: 'Segoe UI', sans-serif;
    }
    QWidget#MainForm {
        background-color: #f7f9fc;
    }
    QFrame#Card {
        background-color: white;
        border-radius: 12px;
        border: 1px solid #e1e7ef;
    }
    QLabel#Title {
        font-size: 26px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    QLineEdit {
        padding: 10px;
        border: 2px solid #e1e7ef;
        border-radius: 6px;
        background: #fdfdfd;
        font-size: 14px;
        margin-bottom: 10px;
    }
    QLineEdit:focus {
        border-color: #3498db;
    }
    QPushButton#PrimaryButton {
        background-color: #3498db;
        color: white;
        border-radius: 6px;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        margin-top: 10px;
    }
    QPushButton#PrimaryButton:hover {
        background-color: #2980b9;
    }
    QPushButton#LinkButton {
        background: transparent;
        color: #3498db;
        border: none;
        font-size: 13px;
        text-decoration: underline;
    }
    QPushButton#LinkButton:hover {
        color: #2980b9;
    }
    QCheckBox {
        color: #666;
        font-size: 13px;
        margin-bottom: 10px;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
    }
    #TitleBar {
        background-color: transparent;
    }
    #SystemButton {
        background: transparent;
        color: #555;
        border: none;
        font-size: 16px;
        min-width: 30px;
        max-width: 30px;
        min-height: 30px;
        max-height: 30px;
        border-radius: 4px;
    }
    #SystemButton:hover {
        background-color: #e0e0e0;
    }
    #CloseButton:hover {
        background-color: #e74c3c;
        color: white;
    }
"""

class AuthWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.CONFIG_FILE = "login_config.json"
        # 移除默认边框并设置透明背景以便自定义圆角
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMinMaxButtonsHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self._drag_pos = None
        self.init_ui()
        self.setStyleSheet(STYLING)
        
    def init_ui(self):
        self.setFixedSize(450, 600)
        
        # 最外层布局，用于给窗口留出阴影/圆角空间（如果需要）
        self.container_layout = QVBoxLayout(self)
        self.container_layout.setContentsMargins(10, 10, 10, 10) # 这里的边距可以给边框留白，如果想要完全充满请设为0
        
        # 真正的背景容器，设置 ID 以便应用样式中的背景色和圆角
        self.main_content = QFrame()
        self.main_content.setObjectName("MainForm")
        self.main_content.setStyleSheet("#MainForm { background-color: #f7f9fc; border-radius: 15px; }")
        self.container_layout.addWidget(self.main_content)

        # 内部主布局
        self.main_v_layout = QVBoxLayout(self.main_content)
        self.main_v_layout.setContentsMargins(0, 0, 0, 0)
        self.main_v_layout.setSpacing(0)

        # 自定义标题栏
        self.title_bar = QWidget()
        self.title_bar.setObjectName("TitleBar")
        self.title_bar.setFixedHeight(40)
        self.title_layout = QHBoxLayout(self.title_bar)
        self.title_layout.setContentsMargins(15, 5, 5, 0)
        
        # 标题文字
        title_label = QLabel("道路病害分割系统")
        title_label.setStyleSheet("color: #666; font-size: 12px; font-weight: bold;")
        self.title_layout.addWidget(title_label)
        
        self.title_layout.addStretch()

        self.min_btn = QPushButton("—")
        self.min_btn.setObjectName("SystemButton")
        self.min_btn.clicked.connect(self.showMinimized)

        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("SystemButton")
        self.close_btn.setObjectName("CloseButton")
        self.close_btn.setStyleSheet("#CloseButton:hover { background-color: #e74c3c; color: white; border-top-right-radius: 15px; }")
        self.close_btn.clicked.connect(self.close)

        self.title_layout.addWidget(self.min_btn)
        self.title_layout.addWidget(self.close_btn)
        self.main_v_layout.addWidget(self.title_bar)
        
        # 内容区域
        self.content_container = QWidget()
        self.main_layout = QVBoxLayout(self.content_container)
        self.main_layout.setAlignment(Qt.AlignCenter)
        self.main_layout.setContentsMargins(0, 0, 0, 20)
        
        # 卡片容器
        self.card = QFrame()
        self.card.setObjectName("Card")
        self.card_layout = QVBoxLayout(self.card)
        self.card_layout.setContentsMargins(40, 50, 40, 50)
        self.main_layout.addWidget(self.card)

        self.main_v_layout.addWidget(self.content_container)
        
        # 页面切换逻辑
        self.show_login()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if Qt.LeftButton and self._drag_pos:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def clear_card(self):
        """清除卡片中的所有控件"""
        while self.card_layout.count():
            item = self.card_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def show_login(self):
        self.clear_card()
        
        # 加载记住的信息
        saved_info = self.load_login_info()
        
        title = QLabel("欢迎使用道路病害分割系统")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        self.card_layout.addWidget(title)
        
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("请输入用户名")
        self.user_input.setText(saved_info.get("username", ""))
        self.card_layout.addWidget(self.user_input)
        
        self.pwd_input = QLineEdit()
        self.pwd_input.setPlaceholderText("请输入密码")
        self.pwd_input.setEchoMode(QLineEdit.Password)
        self.pwd_input.setText(saved_info.get("password", ""))
        self.card_layout.addWidget(self.pwd_input)
        
        self.remember_checkbox = QCheckBox("记住密码")
        self.remember_checkbox.setChecked(saved_info.get("remember", False))
        self.card_layout.addWidget(self.remember_checkbox)
        
        login_btn = QPushButton("登 录")
        login_btn.setObjectName("PrimaryButton")
        login_btn.clicked.connect(self.handle_login)
        self.card_layout.addWidget(login_btn)
        
        reg_link = QPushButton("没有账号？立即注册")
        reg_link.setObjectName("LinkButton")
        reg_link.clicked.connect(self.show_register)
        self.card_layout.addWidget(reg_link, alignment=Qt.AlignCenter)

    def load_login_info(self):
        """从本地文件加载保存的登录信息"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载登录配置失败: {e}")
        return {"username": "", "password": "", "remember": False}

    def save_login_info(self, username, password, remember):
        """保存登录信息到本地文件"""
        data = {
            "username": username,
            "password": password if remember else "",
            "remember": remember
        }
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存登录配置失败: {e}")

    def show_register(self):
        self.clear_card()
        
        title = QLabel("新用户注册")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        self.card_layout.addWidget(title)
        
        self.reg_user_input = QLineEdit()
        self.reg_user_input.setPlaceholderText("选择一个专属用户名")
        self.card_layout.addWidget(self.reg_user_input)
        
        self.reg_pwd_input = QLineEdit()
        self.reg_pwd_input.setPlaceholderText("设置您的密码")
        self.reg_pwd_input.setEchoMode(QLineEdit.Password)
        self.card_layout.addWidget(self.reg_pwd_input)
        
        self.reg_pwd_confirm = QLineEdit()
        self.reg_pwd_confirm.setPlaceholderText("确认您的密码")
        self.reg_pwd_confirm.setEchoMode(QLineEdit.Password)
        self.card_layout.addWidget(self.reg_pwd_confirm)
        
        reg_btn = QPushButton("注 册")
        reg_btn.setObjectName("PrimaryButton")
        reg_btn.clicked.connect(self.handle_register)
        self.card_layout.addWidget(reg_btn)
        
        login_link = QPushButton("已有账号？返回登录")
        login_link.setObjectName("LinkButton")
        login_link.clicked.connect(self.show_login)
        self.card_layout.addWidget(login_link, alignment=Qt.AlignCenter)

    def handle_login(self):
        user = self.user_input.text()
        pwd = self.pwd_input.text()
        remember = self.remember_checkbox.isChecked()
        
        if not user or not pwd:
            QMessageBox.warning(self, "认证检查", "用户名和密码均不能为空！")
            return
            
        success, msg = DBManager.login_user(user, pwd)
        if success:
            # 登录成功，保存记住的信息
            self.save_login_info(user, pwd, remember)
            
            QMessageBox.information(self, "认证成功", f"欢迎回来, {user}!")
            # 登录成功，跳转到主界面
            from main_window import MainWindow
            self.mw = MainWindow(username=user)
            self.mw.show()
            self.close() 
        else:
            QMessageBox.critical(self, "认证失败", msg)

    def handle_register(self):
        user = self.reg_user_input.text()
        pwd = self.reg_pwd_input.text()
        pwd_confirm = self.reg_pwd_confirm.text()
        
        if not user or not pwd:
            QMessageBox.warning(self, "输入错误", "用户名和密码均不能为空！")
            return
        
        if pwd != pwd_confirm:
            QMessageBox.warning(self, "输入错误", "两次填写的密码不一致")
            return
            
        success, msg = DBManager.register_user(user, pwd)
        if success:
            QMessageBox.information(self, "成功", msg)
            self.show_login()
        else:
            QMessageBox.critical(self, "失败", msg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AuthWindow()
    window.show()
    sys.exit(app.exec_())
