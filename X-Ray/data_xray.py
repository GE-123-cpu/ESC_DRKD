import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_mask = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor()
])


class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, image_folder):
        self.csv_file = csv_file
        self.image_folder = image_folder
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # 只保留has_pneumo为0的行
        self.data = self.data[self.data['has_pneumo'] == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")  # 确保图像是RGB格式

        if self.transform:
            image = self.transform(image)

        label = self.data.iloc[idx, 2]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255., (256, 256))

        return image, img, label


class TestPneumoniaDataset(Dataset):
    def __init__(self, csv_file, image_folder, mask_image_folder):
        self.csv_file = csv_file
        self.image_folder = image_folder
        self.data = pd.read_csv(csv_file)
        self.mask_image_folder = mask_image_folder
        self.transform = transform
        self.mask_transform = transform_mask
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        mask_img_name = os.path.join(self.mask_image_folder, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_img_name).convert("L")
        image = self.transform(image)
        mask = self.mask_transform(mask)
        label = self.data.iloc[idx, 2]
        return image, mask, label

