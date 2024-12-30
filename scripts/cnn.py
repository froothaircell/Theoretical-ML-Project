import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    Simple CNN for bounding box regression (Algorithm 2).
    Uses convolution + pooling, then flattens to a small MLP head.
    Output: 4D bounding box [x, y, w, h].
    """

    def __init__(self):
        super(CNNModel, self).__init__()

        # Encoder (3 convolution layers)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Regressor head
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Global average pooling
        x = self.global_pool(x)  # shape: (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)

        # Fully connected for bounding box
        x = F.relu(self.fc1(x))
        out = self.fc2(x)  # (B, 4)
        return out
