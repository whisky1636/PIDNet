import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 读取图像
img_path = "gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png"

# 方法1：使用OpenCV
img_cv = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 保持原始格式
print(f"OpenCV读取 - 形状: {img_cv.shape}")
print(f"数据类型: {img_cv.dtype}")
print(f"唯一值（掩码ID）: {np.unique(img_cv)}")
print(f"值范围: {img_cv.min()} - {img_cv.max()}")

# 方法2：使用PIL
img_pil = Image.open(img_path)
img_array = np.array(img_pil)
print(f"\nPIL读取 - 形状: {img_array.shape}")
print(f"数据类型: {img_array.dtype}")
print(f"唯一值（掩码ID）: {np.unique(img_array)}")

# 统计每个掩码的数量
unique, counts = np.unique(img_array, return_counts=True)
print("\n掩码统计:")
for mask_id, count in zip(unique, counts):
    print(f"  掩码ID {mask_id}: {count} 像素 ({count/img_array.size*100:.2f}%)")