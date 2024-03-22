"""
@brief Classify a folder of images using the best model
"""

import os
import sys
import torch
import torchvision
import json

from torchvision import transforms
from PIL import Image

# Import the main CNN model from another file
sys.path.append(os.path.abspath('../model'))
from model import CNN

# Dictionary mapping class numbers to class labels
class_labels = {
    0: "2-4-Time",
    1: "2-8-Time", 
    2: "3-4-Time", 
    3: "3-8-Time",
    4: "4-4-Time",
    5: "5-4-Time",
    6: "6-8-Time",
    7: "Accent",
    8: "Barline",
    9: "Beam",
    10: "Cut-Time",
    11: "Dot",
    12: "Eighth-Note",
    13: "Eighth-Rest",
    14: "F-Clef",
    15: "Fermata",
    16: "Flat",
    17: "G-Clef",
    18: "Half-Note",
    19: "Marcato",
    20: "Mordent",
    21: "Multiple-Half-Notes",
    22: "Natural",
    23: "Quarter-Note",
    24: "Quarter-Rest",
    25: "Sharp",
    26: "Sixteeth-Note",
    27: "Sixteeth-Rest",
    28: "Sixty-Four-Note",
    29: "Sixty-Four-Rest",
    30: "Tenuto",
    31: "Thirty-Two-Note",
    32: "Thirty-Two-Rest",
    33: "Whole-Half-Rest",
    34: "Whole-Note"
}

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
transform = transforms.Compose([
    transforms.Grayscale(), 
    transforms.Resize((224,224)), 
    transforms.ToTensor()
])

# Get the path containing images to classify
while True:
    img_dir = input("Enter directory path to be classified: ")
    if not img_dir or not os.path.isdir(img_dir):
        print("Invalid input, try again")
    else:
        break

# Path to write data to
while True:
    output_path = input("Enter folder path where output will be stored: ")
    if not output_path or not os.path.isdir(output_path):
        print("Invalid input, try again")
    else:
        break

# Loop through each image in folder and get prediction
predictions = []
for image_name in sorted(os.listdir(img_dir)):

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

predicted_labels = [class_labels.get(class_number, "Unknown") for class_number in predictions]
text_string = ':'.join(predicted_labels)
text_string = f'<{text_string}>'

# Write data to json file
data = {"classification": text_string}
filename = 'symbols.json'
json_path = f'{output_path}/{filename}'
with open(json_path, 'w') as file:
    json.dump(data, file)