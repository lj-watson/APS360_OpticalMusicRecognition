"""
@brief The CNN model

Last updated: 03/07/24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layer 1: 1 input channel (BW), 16 filters (output channels), kernel size of 5x5
        self.name = "OMR_CNN"
        self.conv1 = nn.Conv2d(1, 16, 5)
        # Max pooling layer with 2x2 window, with stride of 2 (non-overlapping pooling)
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization after conv1
        self.pool = nn.MaxPool2d(2, 2)
        # 2nd Convolutional layer
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization after conv1
        # Fully connected network
        self.fc1 = nn.Linear(54 * 54 * 32, 64)
        self.fc2 = nn.Linear(64, 36)
        self.dropout = nn.Dropout(p=0.5)  # Add dropout layer with dropout probability 0.5

    def forward(self, x):
        # Pass through convolutional and pooling layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 54 * 54 * 32)
        # ReLU activation function
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first fully connected layer
        x = self.fc2(x)
        # Using cross entropy loss, so we don't need to apply softmax
        return x