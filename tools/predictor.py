import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os
import sys
import time

# 确保可以导入 models 文件夹下的内容
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pidnet import PIDNet

class RoadDamagePredictor:
    def __init__(self, model_path='models/best.pt', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = self._load_model()
        
        # 调试输出
        print(f"Model loaded on {self.device}")

    def _load_model(self):
        # 根据常规 PIDNet 设置初始化，如果需要根据 yaml 动态配置可后续修改
        # 假设使用的是 pidnet-s 或根据 best.pt 推断的结构
        model = PIDNet(m=2, n=3, num_classes=2, planes=32, ppm_planes=96, head_planes=128, augment=False)
        
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            # 处理 "model." 前缀并跳过不匹配的键
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[6:] if k.startswith('model.') else k  # 去掉 'model.' 前缀
                new_state_dict[name] = v
            
            # 使用 strict=False 加载以忽略 num_batches_tracked 等多余键
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded checkpoint with results: {msg}")
        else:
            print(f"Warning: {self.model_path} not found!")
            
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, img_path):
        """图像预处理"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 标准化和缩放逻辑 (根据训练时的设置，此处暂定常规操作)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        h, w, _ = img.shape
        # 假设输入尺寸
        input_size = (1024, 1024) 
        img_resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
        
        img_normalized = (img_resized / 255.0 - mean) / std
        img_input = img_normalized.transpose(2, 0, 1) # HWC to CHW
        img_input = torch.from_numpy(img_input).unsqueeze(0).float()
        
        return img_input.to(self.device), (h, w)

    def predict(self, img_path_or_array):
        """执行推理并返回分割图"""
        if isinstance(img_path_or_array, str):
            input_tensor, ori_size = self.preprocess(img_path_or_array)
        else:
            # 处理 numpy 数组 (视频帧)
            input_tensor, ori_size = self.preprocess_array(img_path_or_array)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            # PIDNet 输出通常包含多个辅助输出，取主输出
            if isinstance(output, (list, tuple)):
                output = output[1] # 通常 index 1 是主分割输出
            
            output = F.interpolate(output, size=ori_size, mode='bilinear', align_corners=True)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
        return prediction

    def preprocess_array(self, img):
        """对 numpy 数组进行预处理"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 标准化和缩放逻辑
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        h, w, _ = img.shape
        input_size = (1024, 1024) 
        img_resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
        
        img_normalized = (img_resized / 255.0 - mean) / std
        img_input = img_normalized.transpose(2, 0, 1) # HWC to CHW
        img_input = torch.from_numpy(img_input).unsqueeze(0).float()
        
        return img_input.to(self.device), (h, w)

    def predict_video(self, video_path, base_save_dir, progress_callback=None):
        """识别视频并保存"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "无法打开视频文件"

        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建保存目录
        video_name = os.path.basename(os.path.normpath(video_path))
        video_name_no_ext = os.path.splitext(video_name)[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        target_save_path = os.path.join(base_save_dir, f"result_{video_name_no_ext}_{timestamp}.mp4")

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(target_save_path, fourcc, fps, (width, height))

        # 调色板定义
        palette = np.zeros((2, 3), dtype=np.uint8)
        palette[0] = [0, 0, 0]        # 背景
        palette[1] = [255, 255, 255]  # 病害

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 推理
            prediction = self.predict(frame)
            
            # 将预测结果转换为 BGR 格式以便保存到视频 (OpenCV 使用 BGR)
            # prediction 是包含 0, 1 的数组
            res_frame = palette[prediction.astype(np.uint8)]
            # 转换为 BGR (如果伪彩色定义是 RGB)
            res_frame = cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR)
            
            out.write(res_frame)
            
            frame_count += 1
            if progress_callback:
                progress_callback(frame_count, total_frames)

        cap.release()
        out.release()
        
        return target_save_path, frame_count

    def save_result(self, prediction, save_path):
        """将预测结果转换为伪彩色图并保存"""
        # 定义颜色映射 (0: 背景-黑色, 1: 病害-白色)
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[0] = [0, 0, 0]
        palette[1] = [255, 255, 255]
        
        # 应用调色板
        pred_img = Image.fromarray(prediction.astype(np.uint8), mode='P')
        pred_img.putpalette(palette.flatten())
        pred_img.convert('RGB').save(save_path)
        return save_path

    def predict_folder(self, folder_path, base_save_dir, progress_callback=None):
        """识别整个文件夹中的图片"""
        # 支持的图片格式
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        
        if not img_files:
            return None, "文件夹中没有有效的图片文件"

        # 创建保存目录：(用户上传的目录最后一级 + 时间)
        folder_name = os.path.basename(os.path.normpath(folder_path))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        target_save_dir = os.path.join(base_save_dir, f"{folder_name}_{timestamp}")
        
        if not os.path.exists(target_save_dir):
            os.makedirs(target_save_dir)

        results = []
        for i, img_name in enumerate(img_files):
            img_path = os.path.join(folder_path, img_name)
            try:
                prediction = self.predict(img_path)
                save_path = os.path.join(target_save_dir, f"result_{img_name}")
                self.save_result(prediction, save_path)
                results.append(save_path)
            except Exception as e:
                print(f"处理图片 {img_name} 时出错: {e}")
            
            if progress_callback:
                progress_callback(i + 1, len(img_files))
        
        return target_save_dir, len(results)
