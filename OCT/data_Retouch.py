import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
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







class RetouchFolder(Dataset):
    def __init__(self, root_dir, transform = transform, transform_mask = transform_mask):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'image')
        self.label_dir = os.path.join(root_dir, 'label')
        self.transform = transform
        self.transform_mask = transform_mask
        self.image_paths = []
        self.label_paths = []

        # Populate image_paths and label_paths lists
        for train_folder in os.listdir(self.img_dir):
            img_folder_path = os.path.join(self.img_dir, train_folder)
            label_folder_path = os.path.join(self.label_dir, train_folder)

            if os.path.isdir(img_folder_path) and os.path.isdir(label_folder_path):
                for img_file in os.listdir(img_folder_path):
                    img_path = os.path.join(img_folder_path, img_file)
                    label_path = os.path.join(label_folder_path, img_file.replace('image', 'label'))
                    self.image_paths.append(img_path)
                    self.label_paths.append(label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
            label = self.transform_mask(label)

        label_sum = torch.sum(label)
        label_value = 0 if label_sum == 0 else 1
        label[label > 0] = 1
        label[label <= 0] = 0

        return image, label_value, label