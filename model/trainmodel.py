"""
@brief Program to train the CNN model

Last updated: 03/08/24
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

        imgs = imgs.to(device)
        labels = labels.to(device)

        output = model(imgs)

        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def get_loss(model, loader, criterion):
    total_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    loss = float(total_loss) / (i + 1)
    return loss

def early_stopping(val_loss, min_val_loss, patience, min_delta, counter):
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        counter = 0
    elif val_loss > (min_val_loss + min_delta):
        counter += 1
        if counter >= patience:
            return True, min_val_loss, counter
    return False, min_val_loss, counter

def train(model, train_data, val_data, batch_size=32, learning_rate=0.01, num_epochs=1, patience = 3, min_delta=10):

    # Initialize minimum validation loss for early stopping
    min_val_loss = float('inf')
    counter = 0

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle = True)
    criterion = nn.CrossEntropyLoss()
    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0004)
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, weight_decay= 0.001, momentum = 0.9)

    train_acc = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    # training
    print("Starting training")
    start_time = time.time()
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):

            imgs = imgs.to(device)
            labels = labels.to(device)

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

        # Check early stopping
        early_stop, min_val_loss, counter = early_stopping(val_loss[epoch], min_val_loss, patience, min_delta, counter)
        if early_stop:
            break

    print("Finished training")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    plt.figure()
    plt.title("Training vs Validation Accuracy")
    plt.plot(range(num_epochs), train_acc, label="Train")
    plt.plot(range(num_epochs), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.ylim(0.8, 1)  # Set y-axis limits from 0 to 1
    plt.savefig("training_validation_acc.png")

    plt.figure()
    plt.title("Training vs Validation Loss")
    plt.plot(range(num_epochs), train_loss, label="Train")
    plt.plot(range(num_epochs), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.ylim(0, 0.5)  # Set y-axis limits from 0 to 1
    plt.savefig("training_validation_loss.png")

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
    # Image folder by default loads 3 colour channels, so transform to grayscale
    #transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=data['mean'], std=data['std'])])
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    train_dataset = ImageFolder(os.path.join(dataset_path, "train"), transform=transform)
    val_dataset = ImageFolder(os.path.join(dataset_path, "val"), transform=transform)

    model = CNN()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    train(model, train_dataset, val_dataset, batch_size=16, learning_rate=0.003, num_epochs=15)
