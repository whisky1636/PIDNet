# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class Crack500(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=2,
                 multi_scale=True,
                 flip=True, 
                 ignore_label=-1,
                 base_size=640,
                 crop_size=(640, 640),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(Crack500, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()

        self.label_mapping = {0: 0, 255: 1}
        self.class_weights = torch.FloatTensor([0.9331, 1.0669]).cuda()
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    # def __getitem__(self, index):
    #     item = self.files[index]
    #     name = item["name"]
    #     image = cv2.imread(os.path.join(self.root,  item["img"]),
    #                        cv2.IMREAD_COLOR)
    #     size = image.shape
    #
    #     if 'test' in self.list_path:
    #         image = self.input_transform(image)
    #         image = image.transpose((2, 0, 1))
    #
    #         return image.copy(), np.array(size), name
    #
    #     label = cv2.imread(os.path.join(self.root,  item["label"]),
    #                        cv2.IMREAD_GRAYSCALE)
    #     label = self.convert_label(label)
    #
    #     image, label, edge = self.gen_sample(image, label,
    #                                          self.multi_scale, self.flip, edge_size=self.bd_dilate_size)
    #
    #     return image.copy(), label.copy(), edge.copy(), np.array(size), name
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, item["img"]), cv2.IMREAD_COLOR)
        size = image.shape

        # 1. 处理测试模式 (test.lst)
        if 'test' in self.list_path:
            # 强制缩放到配置尺寸 [512, 512]
            image = cv2.resize(image, (self.crop_size[1], self.crop_size[0]),
                               interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.array(size), name

        # 2. 读取并转换标签
        label = cv2.imread(os.path.join(self.root, item["label"]), cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        # 3. 处理验证模式 (val.lst 或 trainval.lst)
        # 如果不是训练模式，或者路径里明确带有 val
        if 'train' not in self.list_path or 'val' in self.list_path:
            target_h, target_w = self.crop_size  # 即 512, 512

            # 统一尺寸
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            # --- 关键：手动生成 Edge ---
            # 既然没有 get_edge，我们使用 cv2 的 Canny 或 Laplacian 模拟 PIDNet 的边界提取
            # PIDNet 的边缘通常是标签的 Laplacian 之后进行膨胀
            edge = cv2.Laplacian(label, cv2.CV_64F)
            edge = np.where(np.abs(edge) > 0, 1, 0).astype(np.uint8)
            # 膨胀处理，使其边缘变粗，符合 bd_dilate_size
            if self.bd_dilate_size > 0:
                kernel = np.ones((self.bd_dilate_size, self.bd_dilate_size), np.uint8)
                edge = cv2.dilate(edge, kernel, iterations=1)

            # 图像标准化
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), edge.copy(), np.array(size), name

        # 4. 训练模式 (保持原样，gen_sample 会内部处理 edge)
        image, label, edge = self.gen_sample(image, label,
                                             self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred



    def save_single_image(pred, sv_path, name, label_mapping, ignore_label):
        """
        单张图像保存逻辑，封装以便多线程调用
        """
        # 向量化标签转换逻辑 (inverse=True)
        # 原 self.convert_label 内部通常是循环，这里直接用 numpy 映射或布尔索引加速
        temp = pred.copy()
        for k, v in label_mapping.items():
            pred[temp == v] = k

        # 转换为 uint8 并保存
        save_img = Image.fromarray(pred.astype(np.uint8))
        save_img.save(os.path.join(sv_path, name + '.png'))

    def save_pred(self, preds, sv_path, name):
        """
        批量保存预测结果图片
        """
        # 1. 在 GPU 上完成 argmax (得到 0 和 1)
        preds_idx = preds.argmax(dim=1).cpu().numpy().astype(np.uint8)

        # 2. 准备映射表 (把 1 映射为 255)
        # 你的 self.label_mapping 是 {0: 0, 255: 1}
        # 逆向映射应该是 {0: 0, 1: 255}
        lut = np.zeros(256, dtype=np.uint8)
        for k, v in self.label_mapping.items():
            lut[v] = k  # lut[0]=0, lut[1]=255

        for i in range(preds_idx.shape[0]):
            # 3. 使用 LUT 映射，将 1 变为 255
            pred_converted = lut[preds_idx[i]]

            # 4. 创建灰度图 (L 模式) 直接保存
            # 此时 0 是全黑，255 是全白，视觉上就能看清了
            save_img = Image.fromarray(pred_converted, mode='L')

            # 确保保存路径存在
            if not os.path.exists(sv_path):
                os.makedirs(sv_path)

            save_img.save(os.path.join(sv_path, name[i] + '.png'))

        
        
