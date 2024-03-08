"""
@brief Program to train the CNN model

Last updated: 03/07/24
"""

import json
import sys
import os
import torch
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CNN

# Get the directory of dataset
def get_directory_path():
    while True:
        directory_path = input("Enter directory path of dataset: ")
        if not directory_path or not os.path.isdir(directory_path):
            print("Invalid input, try again")
        else:
            return directory_path
        
def get_model_name(name, batch_size, learning_rate, epoch):
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

def get_accuracy(model, data):

    correct = 0
    total = 0
    for imgs, labels in data:

        output = model(imgs)

        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def train(model, train_data, val_data, batch_size=32, learning_rate=0.01, num_epochs=1):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle = True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_acc = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)

    # training
    print("Starting training")
    start_time = time.time()
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):

            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

        # save the current training information
        train_acc[epoch] = (get_accuracy(model, train_loader)) # compute training accuracy
        val_acc[epoch] = (get_accuracy(model, val_loader))  # compute validation accuracy

        print(("Epoch {}: Train accuracy: {} |"+
               "Validation accuracy: {}").format(
                   epoch + 1,
                   train_acc[epoch],
                   val_acc[epoch]))
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
        torch.save(model.state_dict(), model_path)

    print("Finished training")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    plt.title("Training Curve")
    plt.plot(range(num_epochs), train_acc, label="Train")
    plt.plot(range(num_epochs), val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":

    torch.manual_seed(42)

    meanstd_path = 'meanstd.json'
    try:
        with open(meanstd_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Could not find file containing mean and std of dataset")
        sys.exit(1)

    dataset_path = get_directory_path()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((data['mean'],), (data['std'],))])
    train_dataset = ImageFolder(os.path.join(dataset_path, "train"), transform=transform)
    val_dataset = ImageFolder(os.path.join(dataset_path, "val"), transform=transform)
    test_dataset = ImageFolder(os.path.join(dataset_path, "test"), transform=transform)

    OMR_CNN = CNN()
    train(OMR_CNN, train_dataset, val_dataset, batch_size=64, learning_rate=0.003, num_epochs=40)
