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
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

transform_mask = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
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


class test_fluidFolder(Dataset):
    def __init__(self, nor_image_dir, abnor_image_dir, mask_dir):
        self.nor_image_dir = nor_image_dir
        self.abnor_img_dir = abnor_image_dir
        self.m_dir = mask_dir
        self.transform_img = transform_mask
        self.transform_mask = transform_mask
        self.nor_list = os.listdir(nor_image_dir)
        self.abnor_list = os.listdir(abnor_image_dir)
        self.mask_list = os.listdir(mask_dir)

    def __len__(self):
        return len(self.nor_list) + len(self.abnor_list)

    def __getitem__(self, index):
        if index < len(self.nor_list):  # 加载正常图像
            image = Image.open(os.path.join(self.nor_image_dir, self.nor_list[index])).convert('RGB')
            label = 0
            mask = torch.zeros([1, 256, 256])
        else:  # 加载异常图像
            index -= len(self.nor_list)
            image = Image.open(os.path.join(self.abnor_img_dir, self.abnor_list[index])).convert('RGB')
            label = 1
            mask = Image.open(os.path.join(self.m_dir, self.mask_list[index])).convert('L')
            mask = self.transform_mask(mask)


        if self.transform_img:
            image = self.transform_img(image)
        return image, label, mask