"""
@brief Program to prepare the dataset for training

Last updated: 03/07/24
"""

import os
import sys
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
from PIL import Image

# CONSTANTS
VARIABILITY = 50
SET_WIDTH = 224
SET_HEIGHT = 224

# Set a seed
random.seed(10)

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

def add_noise(img):
    deviation = VARIABILITY * random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0, 255)
    return img

def data_augmentation(directory):

    datagen = ImageDataGenerator(
        preprocessing_function=add_noise
        #rotation_range=15,
        #width_shift_range=0.5,
        #height_shift_range=0.5,
        #shear_range=0.05,
        #fill_mode='nearest',
        #channel_shift_range=20,
        #brightness_range=[0.8, 1.2]
    )

    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]

    for subfolder in tqdm(subfolders):
        items = os.listdir(subfolder)
        item_count = len(items)
        folder_path = os.path.join(directory, subfolder)

        i = 0
        for item in items:
            rand = random.random()

            if rand < 0.25:

                img_path = os.path.join(folder_path, items[i])

                curr_img = image.load_img(img_path)

                img_list = [curr_img]
                arr = image.img_to_array(curr_img)
                arr = arr.reshape((1,) + arr.shape)

                for k, batch in enumerate(datagen.flow(arr, batch_size=1)):
                    new_img = image.array_to_img(batch[0])
                    img_list.append(new_img)

                    save_path = os.path.join(folder_path, "aug_{}_{}.png".format(item, k))
                    new_img.save(save_path)

                    if len(img_list) >= 4:
                        break

                #rows, cols = 2, 2
                #fig, array = plt.subplots(rows, cols)

                #for j in range(4):
                    #array[j // cols, j % cols].imshow(img_list[j])

                #plt.show()

            else:
                continue

            i += 1
            if (i > item_count - 1):
                break

if __name__ == "__main__":

    # Get the path to the dataset to train on
    dataset_path = get_directory_path()

    # Resize all images
    for(root, dirs, files) in tqdm(os.walk(dataset_path), desc="Resizing images"):
        for file in files:
            resize(os.path.join(root, file), SET_WIDTH, SET_HEIGHT)
    
    while True:
        normalize_yn = input("Calculate dataset normalization? (y/n) ").lower()
        if normalize_yn == 'y' or normalize_yn == 'n':
            break
        else:
            continue

    while True:
        augment_yn = input("Augment/Delete augmented data? (y/delete/none) ").lower()
        if augment_yn == 'y' or augment_yn == 'none' or augment_yn == 'delete':
            break
        else:
            continue
            
    if normalize_yn == 'y':
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

        print(f"Writing mean {mean} and std {std} to file...", end="")
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

    if augment_yn == 'y':
        train_path = os.path.join(dataset_path, "train")
        num_files = sum(len(files) for _, _, files in os.walk(train_path))
        print(f"Number of training files is {num_files}. Augmenting data...")
        data_augmentation(train_path)
        print("Done.")
        num_files = sum(len(files) for _, _, files in os.walk(train_path))
        print(f"Number of training files is now {num_files}")
    elif augment_yn == 'delete':
        print("Deleting augmented data...")
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.startswith("aug"):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
        print("Done.")