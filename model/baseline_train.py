"""
@brief Program to train the CNN model

Last updated: 03/08/24
"""

import json
import sys
import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from baseline import LeNet5

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

def get_loss(model, loader, criterion):
    total_loss = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_epoch += len(labels)
    loss = float(total_loss) / (i + 1)
    return loss

def train(model, train_data, val_data, batch_size=32, learning_rate=0.01, num_epochs=1):

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle = True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_acc = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    print("Starting training...")
    start_time = time.time()
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):

            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################

            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

        # save the current training information
        train_acc[epoch] = (get_accuracy(model, train_loader)) # compute training accuracy
        val_acc[epoch] = (get_accuracy(model, val_loader))  # compute validation accuracy
        train_loss[epoch] = get_loss(model, train_loader, criterion) # compute training loss
        val_loss[epoch] = get_loss(model, val_loader, criterion) # compute validation loss

        print(("Epoch {}: Train accuracy: {} | "+
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

    plt.title("Training vs Validation Accuracy")
    plt.plot(range(num_epochs), train_acc, label="Train")
    plt.plot(range(num_epochs), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training vs Validation Loss")
    plt.plot(range(num_epochs), train_loss, label="Train")
    plt.plot(range(num_epochs), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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
    ''', transforms.Normalize(mean=data['mean'], std=data['std'])'''
    # Image folder by default loads 3 colour channels, so transform to grayscale
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    train_dataset = ImageFolder(os.path.join(dataset_path, "train"), transform=transform)
    val_dataset = ImageFolder(os.path.join(dataset_path, "val"), transform=transform)
    test_dataset = ImageFolder(os.path.join(dataset_path, "test"), transform=transform)

    while True:
        try:
            bs = int(input("Enter Batch Size: "))
            lr = float(input("Enter Learning Rate: "))
            ep = int(input("Enter Number of Epochs: "))
            break
        except ValueError:
            print("Please enter valid integers for Batch Size and Number of Epochs, and a valid float for Learning Rate.")

    while True:
        cuda_input = input("Use CUDA? (y/n) ").lower()
        if cuda_input == 'y':
            use_cuda = True
            break
        elif cuda_input == 'n':
            use_cuda = False
            break
        else:
            continue

    model = LeNet5()
    if use_cuda and torch.cuda.is_available():
        model.cuda()
        print('CUDA is available!  Training on GPU ...')
    else:
        print('CUDA is not available.  Training on CPU ...')

    train(model, train_dataset, val_dataset, batch_size=bs, learning_rate=lr, num_epochs=ep)
