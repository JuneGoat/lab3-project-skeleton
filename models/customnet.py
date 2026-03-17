import torch
from torch import nn

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # 定义卷积层和池化层
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 图像尺寸从 64x64 变为 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 图像尺寸变为 16x16
        )
        self.flatten = nn.Flatten()
        # 定义全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 200) # TinyImageNet 有 200 个类别
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)