import pytorch_lightning as pl
from pytorch_lightning import Trainer

import torch
import torch.nn as nn
from torchsummary import summary
from IPython.display import clear_output
from typing import Dict
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard.summary import hparams


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", stride=1, r=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), padding=padding, stride=stride) if r == True else nn.Identity()
        
    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x2 = self.conv3(x)
        return x1+x2
    
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu = nn.ReLU()
       
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels ,skip_channels,scale=2, kernel_size=3, padding="same", stride=1,last=True):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(scale) if scale > 1 else nn.Identity()
        self.bn = nn.BatchNorm2d(skip_channels)
        self.conv = nn.Conv2d(skip_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()  if last == True else nn.Identity()

    def forward(self, x,skip):
        x = self.pixel_shuffle(x)
        x = torch.cat([x, skip], dim=1)
        x = self.bn(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.relu3(x)
        return x