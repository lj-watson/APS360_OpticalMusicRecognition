from model import CNN
from baseline import LeNet5
from trainmodel import get_model_name, get_accuracy, get_directory_path
from baseline_train import get_model_name, get_accuracy, get_directory_path

import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

while True:
    net_input = input("Which model to test? (main/baseline): ").lower()
    if net_input in ['main', 'baseline']:
        net = CNN() if net_input == 'main' else LeNet5()
        break

while True:
    try:
        bs = int(input("Enter Batch Size used during Training: "))
        lr = float(input("Enter Learning Rate used during Training: "))
        ep = int(input("Enter Number of Epochs used during Training: "))
        break
    except ValueError:
        print("Please enter valid integers for Batch Size and Number of Epochs, and a valid float for Learning Rate.")

dataset_path = get_directory_path()

# Image folder by default loads 3 colour channels, so transform to grayscale
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
test_dataset = ImageFolder(os.path.join(dataset_path, "test"), transform=transform)
model_path = get_model_name(net.name, batch_size=bs, learning_rate=lr, epoch=ep)

state = torch.load(model_path)
net.load_state_dict(state)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle = True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
net = net.to(device)

print("Test Classification Accuracy:", get_accuracy(net, test_loader))