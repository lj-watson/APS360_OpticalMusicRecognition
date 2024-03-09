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
        # In: 1; Out: 6
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 0)

    def forward(self, x):
        
        return x
