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
    model_path = input("Enter model path: ")
    if not model_path:
        print("Invalid input, try again")
    else:
        break

model = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

# Transformations for the image input
transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((224,224)), transforms.ToTensor()])

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

text_string = ' : '.join([f'{value}' for _, value in enumerate(predictions, start=1)])
text_string = f'<{text_string}>'
print(text_string)