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
        # Convolutional layer 1: 1 input channel (BW), 16 filters (output channels), kernel size of 10x10
        self.name = "OMR_CNN"
        self.conv1 = nn.Conv2d(1, 16, 10)
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization after conv1
        # 2nd Convolutional layer
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization after conv1
        # 3rd Convolutional layer
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization after conv1
        # Max pooling layer with 2x2 window, with stride of 2 (non-overlapping pooling)
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected network
        self.fc1 = nn.Linear(24 * 24 * 64, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 35)
        self.dropout = nn.Dropout(p=0.4)  # Add dropout layer

    def forward(self, x):
        # Pass through convolutional and pooling layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 24 * 24 * 64)
        # ReLU activation function
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # Using cross entropy loss, so we don't need to apply softmax
        return x