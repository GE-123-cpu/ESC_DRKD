import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import torch.nn as nn


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def denormalization1(x):
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, feature1, feature2):
        cos = nn.functional.cosine_similarity(feature1, feature2, dim=1)
        ano_map = torch.ones_like(cos) - cos
        loss = (ano_map.view(ano_map.shape[0], -1).mean(-1)).mean()
        return loss

class loss_fucntion(nn.Module):
    def __init__(self):
        super(loss_fucntion, self).__init__()

    def forward(self, a, b):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(a)):
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))

        loss = loss / (len(a))
        return loss

def cut(img, t, b):
    # h, w, c = img.shape
    x = np.random.randint(0, img.shape[1] - t)
    y = np.random.randint(0, img.shape[0] - b)
    if (x - t) % 2 == 1:
        t -= 1
    if (y - b) % 2 == 1:
        b -= 1

    roi = img[y:y + b, x:x + t]
    return roi

def paste_patch(img, patch):
    imgh, imgw, imgc = img.shape
    patchh, patchw, patchc = patch.shape

    patch_h_position = random.randrange(1, round(imgh) - round(patchh) - 1)
    patch_w_position = random.randrange(1, round(imgw) - round(patchw) - 1)
    pasteimg = np.copy(img)
    pasteimg[patch_h_position:patch_h_position + patchh, patch_w_position:patch_w_position + patchw, :] = patch + 0.2 * img[patch_h_position:patch_h_position + patchh,
    patch_w_position:patch_w_position + patchw, :]

    return pasteimg

class Normalize(object):
    """
    Only normalize images
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0,1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image


def noise_generate(image_np, t):
    transform = transforms.Compose([
        Normalize(),
        ToTensor(),
    ])
    rotated_list = []
    for i in range(image_np.size(0)):
        np_img = image_np[i]
        patch_img = cut(np_img, t, t)
        patch_img = paste_patch(np_img, patch_img)
        img_noise = transform(patch_img)
        img_noise = torch.unsqueeze(img_noise, dim=0)
        rotated_list.append(img_noise)
    img_noise = torch.cat(rotated_list, dim=0)
    return img_noise

def get_anomap(output, Dn, data, device):
    n, c, h, w = data.shape
    anomaly_map1_kd = torch.ones(1, 64, 64).to(device) - F.cosine_similarity(output[0], Dn[0])
    anomaly_map1_kd = anomaly_map1_kd.unsqueeze(dim=1)
    anomaly_map1_kdx = F.interpolate(anomaly_map1_kd, size=(h, w), mode='bilinear', align_corners=True)

    anomaly_map2_kd = torch.ones(1, 32, 32).to(device) - F.cosine_similarity(output[1], Dn[1])
    anomaly_map2_kd = anomaly_map2_kd.unsqueeze(dim=1)
    anomaly_map2_kdx = F.interpolate(anomaly_map2_kd, size=(h, w), mode='bilinear', align_corners=True)
    # \
    anomaly_map3_kd = torch.ones(1, 16, 16).to(device) - F.cosine_similarity(output[2], Dn[2])
    anomaly_map3_kd = anomaly_map3_kd.unsqueeze(dim=1)
    anomaly_map3_kdx = F.interpolate(anomaly_map3_kd, size=(h, w), mode='bilinear', align_corners=True)  # \
    anomaly_map = (anomaly_map1_kdx + anomaly_map2_kdx + anomaly_map3_kdx) / 3

    return anomaly_map





