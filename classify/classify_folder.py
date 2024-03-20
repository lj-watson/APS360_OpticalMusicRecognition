"""
@brief Classify a folder of images using the best model
"""

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import sys
import os
from PIL import Image
sys.path.append(os.path.abspath('../model'))
from model import CNN
from trainmodel import get_model_name

# Get the model path
while True:
    inp = input("Enter batch size, learning rate, and epoch of model to use: ")
    inputs = inp.split()
    if len(inputs) == 3:
        try:
            bs, lr, ep = map(float, inputs)
            bs, ep = int(bs), int(ep) - 1
            break
        except ValueError:
            print("Invalid input.")
    else:
        print("Invalid input.")

model = CNN()
model_path = get_model_name(model.name, batch_size=bs, learning_rate=lr, epoch=ep)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

# Transformations for the image input
transform = transforms.Compose([transforms.Grayscale(), transforms.Resize(224,224), transforms.ToTensor()])

# Get the path containing images to classify
while True:
    img_dir = input("Enter directory path to be classified: ")
    if not img_dir or not os.path.isdir(img_dir):
        print("Invalid input, try again")
    else:
        break

# Loop through each image in folder and get prediction
predictions = []
for image_name in os.listdir(img_dir):

    img_path = os.path.join(img_dir, image_name)
    if not os.path.isfile(img_path):
        continue
    image = Image.open(img_path)

    tensor_img = transform(image)

    # Add batch dimension
    input_tensor = tensor_img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        predictions.append(pred.item())

print(predictions)