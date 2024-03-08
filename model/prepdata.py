"""
@brief Program to prepare the dataset for training

Last updated: 03/07/24
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
 
SET_WIDTH = 224
SET_HEIGHT = 224

# Get the directory of dataset
def get_directory_path():
    while True:
        directory_path = input("Enter directory path of dataset: ")
        if not directory_path or not os.path.isdir(directory_path):
            print("Invalid input, try again")
        else:
            return directory_path

# Prepare data for training, first resize all images
def resize(img_path, width, height):
    if os.path.isfile(img_path) and img_path.lower().endswith(('.png')):
        # Ensure file is black and white
        try:
            pure_img = Image.open(img_path)
            pure_width, pure_height = pure_img.size
            if pure_width != width or pure_height != height:
                new_img = pure_img.resize((height, width), Image.Resampling.LANCZOS)
                new_img.save(img_path)
        except Exception as e:
            print(f"Error resizing image: {e}")
            sys.exit(1)

if __name__ == "__main__":

    # Get the path to the dataset to train on
    dataset_path = get_directory_path()

    # Resize all images
    for(root, dirs, files) in tqdm(os.walk(dataset_path), desc="Resizing images"):
        for file in files:
            resize(os.path.join(root, file), SET_WIDTH, SET_HEIGHT)
    '''
    # Load the data into Pytorch datset, transform into tensor for model
    # Transform to tensors of normalized range using calculated mean and standard deviation
    dataset_unnormalized = ImageFolder(root=dataset_path, transform=transforms.ToTensor())

    mean = np.zeros(1)
    std = np.zeros(1)
    k = 1
    for image, _ in tqdm(dataset_unnormalized, "Computing mean and std of dataset", len(dataset_unnormalized), unit=" samples"):
        image = np.array(image)
        pixels = image.flatten()

        for pixel in pixels:
            pixel_diff = pixel - mean
            mean += pixel_diff/k
            std += pixel_diff * (pixel - mean)
            k += 1
    std = np.sqrt(std / (k - 2))

    print("Writing new values to file...", end="")
    meanstd_path = 'meanstd.json'
    try:
        with open(meanstd_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    data['mean'] = mean
    data['std'] = std

    with open(meanstd_path, 'w') as file:
        json.dump(data, file, indent=4)
    print("Done.")
    '''