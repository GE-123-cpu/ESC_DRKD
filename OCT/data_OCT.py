from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from pathlib import Path
from torchvision import transforms
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import cv2

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])


class trainFolder(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)


    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_list[index])
        inputs = Image.open(image_path).convert('RGB')
        images = self.transform(inputs)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255., (256, 256))
        return images, img

    def __len__(self):
        return len(self.image_list)


class ImageDataset(Dataset):
    def __init__(self, normal_dir, anomalous_dir1, anomalous_dir2, anomalous_dir3):
        self.normal_dir = normal_dir
        self.anomalous_dir1 = anomalous_dir1
        self.anomalous_dir2 = anomalous_dir2
        self.anomalous_dir3 = anomalous_dir3
        self.transform = transform
        # 获取正常和异常图像文件列表
        self.normal_images = os.listdir(normal_dir)
        self.anomalous_images1 = os.listdir(anomalous_dir1)
        self.anomalous_images2 = os.listdir(anomalous_dir2)
        self.anomalous_images3 = os.listdir(anomalous_dir3)

    def __len__(self):
        return len(self.normal_images) + len(self.anomalous_images1) + len(self.anomalous_images2) + len(self.anomalous_images3)

    def __getitem__(self, index):
        if index < len(self.normal_images):  # 加载正常图像
            image = Image.open(os.path.join(self.normal_dir, self.normal_images[index])).convert('RGB')
            label = 0
        else:  # 加载异常图像
            index -= len(self.normal_images)
            if index < len(self.anomalous_images1):  # 异常路径1
                image = Image.open(
                    os.path.join(self.anomalous_dir1, self.anomalous_images1[index])).convert('RGB')
                label = 1
            elif index < len(self.anomalous_images1) + len(self.anomalous_images2):  # 异常路径2
                index -= len(self.anomalous_images1)
                image = Image.open(
                    os.path.join(self.anomalous_dir2, self.anomalous_images2[index])).convert('RGB')
                label = 1
            else:  # 异常路径3
                index -= len(self.anomalous_images1) + len(self.anomalous_images2)
                image = Image.open(
                    os.path.join(self.anomalous_dir3, self.anomalous_images3[index])).convert('RGB')
                label = 1

        if self.transform:
            image = self.transform(image)  # 返回图像及其标签
        return image, label  # 定义图像预处理
