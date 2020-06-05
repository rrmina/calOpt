import torch
import torch.nn as nn
import torch.nn.functional as F

import math

backwarp_ten

def backwarp(x, flow):
    if str

class BasicNetwork(nn.Sequential):
    def __init__(self):
        super(BasicNetwork, self).__init__()

        self.network = nn.Sequential([
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        ])

    def forward(self, x):
        return self.network(x)

class SPyNet(nn.Module):
    def __init__(SPyNet, self)
        super(SPyNet, self).__init__()

        self.network = nn.ModuleList([
            BasicNetwork for i in range(6) 
        ])

    def forward(self, left, right):
        flow = []

        left = [self.normalize(left)]
        right = [self.normalize(right)]

        for i in range(5):
            if ((left[0].shape[2] > 32) or (left[0].shape[3] > 32)):
                left.insert(0, F.avg_pool2d(input=left[0], kernel_size=2, stride=2, count_include_pad=False))
                right.insert(0, F.avg_pool2d(input=right[0], kernel_size=2, stride=2, count_include_pad=False))
                
        flow = left[0].new_zeros([
            left[0].shape[0], 
            2, 
            int(math.floor(left[0].shape[2] / 2.0)), 
            int(math.floor(left[0].shape[3] / 2.0)) 
        ])

        for i in range(len(left)):
            upsampled = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True)

            if (upsampled.shape[2] != left[i].shape[2]):
                upsampled = F.pad(input=upsampled, pad=[0, 0, 0, 1], mode='replicate')
            if (upsampled.shape[3] != left[i].shape[3]):
                upsampled = F.pad(input=upsampled, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.network[i](torch.cat( left[i], ))

    def normalize(self, x):
        B = (x[:, 0, :, :] - 0.406) / 0.225
        G = (x[:, 1, :, :] - 0.456) / 0.224
        R = (x[:, 2, :, :] - 0.485) / 0.229

        return torch.cat([R,G,B], dim=1)

MODEL_PATH = 'network-sintel-final.pytorch'

def calOpt():
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = SPyNet()

    # Load Weights
    key_weights = torch.load(MODEL_PATH)
    key_weights = {key.replace('module', 'net'): weight for key, weight in key_weights}
    model.load_state_dict(key_weights)
    model.to(device)

    # Load Image

    # Forward

    # Upsample

    # Show Results