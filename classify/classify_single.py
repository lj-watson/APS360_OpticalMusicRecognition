"""
@brief Converts notes to text files

Last updated: 03/20/24
"""

import os
import sys
import torch
import torchvision

from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import the main CNN model from another file
sys.path.append(os.path.abspath('../model'))
from model import CNN
from baseline import LeNet5

# Resizing constant
RESIZE = 224

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

# Checks which model to use
while True:
    main_input = input("Which model to test? (main/baseline): ").lower()
    if main_input in ['main', 'baseline']:
        main_model = CNN() if main_input == 'main' else LeNet5()
        model_path = "/Users/danielyu/Desktop/APS360 GitHub Repo/APS360-OpticalMusicRecognition/classify/model_OMR_CNN_bs16_lr0.003_epoch14"  if main_input == 'main' else "/Users/danielyu/Desktop/APS360 GitHub Repo/APS360-OpticalMusicRecognition/classify/model_LeNet5_Baseline_bs32_lr0.001_epoch9"
        break

# Get the path containing images to classify
while True:
    img_dir = input("Enter directory path to be classified: ")
    if not img_dir or not os.path.isdir(img_dir):
        print("Invalid input, try again")
    else:
        break

# Load the neural network model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state = torch.load(model_path, map_location=device)
main_model.load_state_dict(state)
main_model.eval()

# Transform the input image into grayscale and 224 x 224 
transform = transforms.Compose([
    transforms.Grayscale(), 
    transforms.Resize(RESIZE),
    transforms.ToTensor()
])

# ------------------ TEST ------------------ #
# Change this path for single image testing
img_path = "/Users/danielyu/Desktop/APS360 GitHub Repo/APS360-OpticalMusicRecognition/classify/example/Screenshot 2024-03-20 155207.png"
img = Image.open(img_path)
input_tensor = transform(img)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = main_model(input_batch)
# ------------------ TEST ------------------ #

# Find the predicted class by finding th highest probability for each class
probabilities = torch.nn.functional.softmax(output[0], dim=0)
predicted_class = torch.argmax(probabilities).item()
predicted_acc = probabilities[predicted_class] * 100
predicted_acc_final = predicted_acc.item()

# Convert the class number to class label
predicted_label = class_labels.get(predicted_class, "Unknown")

# Print the results
print("Predicted Class: {} --> {}".format(predicted_class, predicted_label))
print("Predicted Accuracy: {:.5f}%".format(predicted_acc_final))
