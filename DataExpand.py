import torch
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import gaussian_blur
import math
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from PIL import Image
from torchvision import transforms


class WeatherAugmentor:
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cnt = random.random()
        if cnt < 0.25:
            if random.random() < 0.5:
                x = self.add_fog(x)
                x = self.add_rain(x)
            else:
                x = self.add_fog(x)
        elif cnt >= 0.25 and cnt < 0.5:
            x = self.add_rain(x)
            x = self.advanced_brightness_contrast(x)
        elif cnt >= 0.5 and cnt < 0.75:
            x = self.add_snow(x)
        else:
            x = self.advanced_brightness_contrast(x)
        return x

    def add_fog(self, x):
        A = x.mean(dim=(1, 2), keepdim=True).max(dim=0, keepdim=True)[0] + 0.1
        depth = torch.rand(1, *x.shape[1:]).to(x.device)
        depth = gaussian_blur(depth, kernel_size=33, sigma=10)
        t = torch.exp(-1 * (1 + 0.2) * depth)
        return x * t + A * (1 - t)

    def add_rain(self, x):
        # 确保输入是 [C, H, W] 格式
        if x.ndim == 4:
            x = x.squeeze(0)  # 如果多一个批次维度，去掉它
        
        h, w = x.shape[1], x.shape[2]
        if h < 5 or w < 1:
            return x  # 图像太小，跳过雨滴生成
        
        rain_layer = torch.zeros_like(x)
        num_drops = random.randint(3000, 4000)
        
        if random.random() > 0.5:
            slope = int(random.choice([-1, 1])) + random.uniform(-0.3, 0.3)
        else:
            slope = 0
        
        for _ in range(num_drops):
            x1 = random.randint(0, h - 5)
            y1 = random.randint(0, w - 1)
            length = random.randint(20, 50)
            intensity = random.uniform(0.2, 0.5)
            
            for i in range(length):
                y_offset = int(y1 + slope * i)  # 倾斜加扰动
                x_pos = x1 + i
                if 0 <= x_pos < h and 0 <= y_offset < w:
                    rain_layer[:, x_pos, y_offset] = intensity
        
        rain_layer = gaussian_blur(rain_layer, kernel_size=(3, 1), sigma=(0.5, 0.1))
        rainy_image = x * (1 - rain_layer) + rain_layer
        return torch.clamp(rainy_image, 0, 1)

    def add_snow(self, x):
        # 确保输入是 [C, H, W] 格式
        if x.ndim == 4:
            x = x.squeeze(0)
        
        base_snow = torch.exp(torch.randn_like(x) * 0.5) * 0.5
        mask = (torch.rand_like(x) > 0.98).float()
        
        # 处理维度问题
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        elif mask.ndim == 5:
            mask = mask.squeeze(0)  # [1, B, C, H, W] -> [B, C, H, W]
        
        mask = F.avg_pool2d(mask, kernel_size=5, padding=2, stride=1)
        mask = (mask > 0.05).float()
        snow_layer = base_snow * mask
        snow_layer = gaussian_blur(snow_layer, kernel_size=(3, 1), sigma=(0.5, 0.1))
        snowy_image = x + snow_layer
        brightness_boost = 1.0 - x.mean([1, 2], keepdim=True) * 0.5
        snowy_image = snowy_image * brightness_boost
        return torch.clamp(snowy_image, 0, 1)

    def advanced_brightness_contrast(self, x):
        # 确保输入是 [C, H, W] 格式
        if x.ndim == 4:
            x = x.squeeze(0)
        
        brightness = random.uniform(0.4, 0.75)  # 明显调暗
        contrast = random.uniform(1.0, 1.3)     # 保持细节
        gamma = random.uniform(1.2, 1.8)        # 整体变暗

        x = TF.adjust_brightness(x, brightness)
        x = TF.adjust_contrast(x, contrast)
        x = TF.adjust_gamma(x, gamma)
        return torch.clamp(x, 0, 1)

# 配置路径
src_img_dir = 'data_simple\\train\\images'
src_lbl_dir = 'data_simple\\train\\labels'
dst_img_dir = 'data_simple\\weather_aug\\images'
dst_lbl_dir = 'data_simple\\weather_aug\\labels'

# 创建输出目录
os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_lbl_dir, exist_ok=True)

# 初始化天气增强器
augmentor = WeatherAugmentor()

# 获取所有图像文件
image_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 处理每张图像
for img_name in tqdm(image_files, desc="Applying weather augmentations"):
    img_path = os.path.join(src_img_dir, img_name)
    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(src_lbl_dir, f"{base_name}.txt")
    
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        continue
    img = Image.open(img_path).convert("RGB")

# 转 tensor 并归一化到 0~1
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img)
    
    # 为每张图像创建10个增强版本
    for i in range(1):
        # 应用天气增强
        with torch.no_grad():
            aug_img_tensor = augmentor(img_tensor.unsqueeze(0)).squeeze(0)
        
        # 转换回OpenCV格式
        aug_img_np = aug_img_tensor.permute(1, 2, 0).numpy()
        aug_img_np = (aug_img_np * 255).clip(0, 255).astype(np.uint8)
        aug_img_bgr = cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR)
        
        # 生成新文件名
        new_img_name = f"{base_name}_aug{i}.jpg"
        new_label_name = f"{base_name}_aug{i}.txt"
        
        # 保存增强后的图像
        cv2.imwrite(os.path.join(dst_img_dir, new_img_name), aug_img_bgr)
        
        # 复制标签文件
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(dst_lbl_dir, new_label_name))
        else:
            # 如果没有标签文件，创建一个空文件
            with open(os.path.join(dst_lbl_dir, new_label_name), 'w') as f:
                pass

print(f"Weather augmentation complete! Generated {len(image_files) * 10} images.")


