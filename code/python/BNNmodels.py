import torch
import torch.nn as nn
import torch.nn.functional as F

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

from Utils import Scale, Clippy

class BNN_FASHION_CNN(nn.Module):
    def __init__(self, quantMethod):
        super(BNN_FASHION_CNN, self).__init__()
        self.quantization = quantMethod

        self.htanh = nn.Hardtanh()
        self.conv1 = QuantizedConv2d(1, 64, kernel_size=3, padding=1, stride=1, quantization=self.quantization)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        self.conv2 = QuantizedConv2d(64, 64, kernel_size=3, padding=1, stride=1, quantization=self.quantization)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=self.quantization)

        self.fc1 = QuantizedLinear(7*7*64, 2048, quantization=self.quantization)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=self.quantization)

        self.fc2 = QuantizedLinear(2048, 10, quantization=self.quantization)
        self.scale = Scale()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.htanh(x)
        x = self.qact1(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh(x)
        x = self.qact2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.htanh(x)
        x = self.qact2(x)

        x = self.fc2(x)
        x = self.scale(x)

        return x

class BNN_FASHION_FC(nn.Module):
    def __init__(self, quantMethod):
        super(BNN_FASHION_FC, self).__init__()
        self.quantization = quantMethod

        self.htanh = nn.Hardtanh()
        self.flatten = torch.flatten
        self.fcfc1 = QuantizedLinear(28*28, 2048, quantization=self.quantization, layerNr=1)
        self.fcbn1 = nn.BatchNorm1d(2048)
        self.fcqact1 = QuantizedActivation(quantization=self.quantization)

        self.fcfc2 = QuantizedLinear(2048, 2048, quantization=self.quantization, layerNr=2)
        self.fcbn2 = nn.BatchNorm1d(2048)
        self.fcqact2 = QuantizedActivation(quantization=self.quantization)
        self.fcfc3 = QuantizedLinear(2048, 10, quantization=self.quantization, layerNr=3)
        self.scale = Scale(init_value=1e-3)

    def forward(self, x):
        x = self.flatten(x, start_dim=1, end_dim=3)
        x = self.fcfc1(x)
        x = self.fcbn1(x)
        x = self.htanh(x)
        x = self.fcqact1(x)

        x = self.fcfc2(x)
        x = self.fcbn2(x)
        x = self.htanh(x)
        x = self.fcqact2(x)

        x = self.fcfc3(x)
        x = self.scale(x)

        return x
