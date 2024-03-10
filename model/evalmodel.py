from model import CNN
from trainmodel import get_model_name, get_accuracy, get_directory_path

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import os

net = CNN()
dataset_path = get_directory_path()
# Image folder by default loads 3 colour channels, so transform to grayscale
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
test_dataset = ImageFolder(os.path.join(dataset_path, "test"), transform=transform)
model_path = get_model_name(net.name, batch_size=80, learning_rate=0.003, epoch=10)
state = torch.load(model_path)
net.load_state_dict(state)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle = True)
if torch.cuda.is_available():
        device = torch.device("cuda")
else:
    device = torch.device("cpu")
net = net.to(device)
print("Test classification accuracy:", get_accuracy(net, test_loader))