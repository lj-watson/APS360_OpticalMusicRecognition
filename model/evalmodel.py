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
        bs = int(input("Enter Batch Size Used During Training: "))
        lr = float(input("Enter Learning Rate Used During Training: "))
        ep = int(input("Enter Number of Epochs Used During Training: ")) - 1
        break
    except ValueError:
        print("Invalid input.")

dataset_path = get_directory_path()

# Image folder by default loads 3 colour channels, so transform to grayscale
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
test_dataset = ImageFolder(os.path.join(dataset_path, "test"), transform=transform)

model_path = get_model_name(net.name, batch_size=bs, learning_rate=lr, epoch=ep)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state = torch.load(model_path, map_location=device)
net.load_state_dict(state)
net.eval()

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle = True)

print("Test Classification Accuracy:", get_accuracy(net, test_loader))