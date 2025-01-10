# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv5(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 256):
        super(YOLOv5, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        # Define the layers here (This is a simplified version)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        # YOLO detection head (for simplicity, let's assume 3 anchors)
        self.detect = nn.Conv2d(128, num_classes + 5, 1)  # 5 for box info (x, y, w, h, confidence)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.detect(x)
        return x
