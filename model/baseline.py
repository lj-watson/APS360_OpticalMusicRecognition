"""
@brief The baseline model

Last updated: 03/09/24
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
        # In: 16 * 53 * 53; Out: 120
        self.fc1 = nn.Linear(16 * 53 * 53, 120) # Calculated to be 53 x 53 from original 224 x 224
        # In: 120; Out: 84
        self.fc2 = nn.Linear(120, 84)
        # In: 84; Out: 10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.avgpool1(x)
        x = F.tanh(self.conv2(x))
        x = self.avgpool2(x)
        x = x.view(-1, 16 * 53 * 53)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x) # No need for softmax due to the usage of Cross Entropy Loss
        
        return x
