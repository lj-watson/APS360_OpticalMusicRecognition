"""
@brief The baseline model

Last updated: 03/08/24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.name = "LeNet5_Baseline"
        # In: 1; Out: 6; Kernel: 5; Stride: 1; Padding: 0
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 0)
        # Kernel: 2; Stride: 2
        self.avgpool1 = nn.AvgPool2d(2, 2)
        # In: 6; Out: 16; Kernel: 5; Stride: 1; Padding: 0
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        # Kernel: 2; Stride: 2
        self.avgpool2 = nn.AvgPool2d(2, 2)
        # In: 16 * x * x; Out: 120
        self.fc1 = nn.Linear()

    def forward(self, x):
        
        return x
