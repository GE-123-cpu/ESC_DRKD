import torch.nn as nn
import torch
import torch.nn.functional as F

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ProjLayer(nn.Module):
    '''
    inputs: features of encoder block
    outputs: projected features
    '''

    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.mc1 = CentralMaskedConv2d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1)
        self.mc2 = CentralMaskedConv2d(in_c // 2, in_c, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_c, in_c//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_c // 2, in_c // 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_c // 4, in_c // 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_c // 2, out_c, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.LeakyReLU()
        self.norm1 = nn.InstanceNorm2d(in_c // 2)
        self.norm2 = nn.InstanceNorm2d(in_c // 4)
        self.norm3 = nn.InstanceNorm2d(in_c // 2)
        self.norm4 = nn.InstanceNorm2d(in_c)

    def forward(self, x):
        x = self.mc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.mc2(x)
        x = self.norm4(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu(x)
        return x


class MultiProjectionLayer(nn.Module):
    def __init__(self, base=64):
        super(MultiProjectionLayer, self).__init__()
        self.proj_a = ProjLayer(base * 4, base * 4)
        self.proj_b = ProjLayer(base * 8, base * 8)
        self.proj_c = ProjLayer(base * 16, base * 16)

    def forward(self, features):
        return [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2])]

