"""
@brief Program to compile multiple datasets of printed music symbols

Last updated: 03/07/24
"""

import os
import sys
import shutil
import splitfolders
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
# Utilize the omrdatasettools library for processing AudiverisOmr dataset
from omrdatasettools.AudiverisOmrImageGenerator import AudiverisOmrImageGenerator
# Get the ignored and mapped classes for each dataset
from dataset_classes import ignored_classes, mapped_classes

# Links to the datasets we will be downloading and processing, available in convenient zip files
# in the OMR-Datasets collection
# @Source: http://www.inescporto.pt/~arebelo/index.php and http://www.inescporto.pt/~jsc/projects/OMR/
# © A. Rebelo, G. Capela, and J. S. Cardoso, Creative Commons BY-SA 4.0 Deed (https://creativecommons.org/licenses/by-sa/4.0/)
REBELO1_URL = "https://github.com/apacha/OMR-Datasets/releases/download/datasets/Rebelo-Music-Symbol-Dataset1.zip"
REBELO2_URL = "https://github.com/apacha/OMR-Datasets/releases/download/datasets/Rebelo-Music-Symbol-Dataset2.zip"
# @Source: https://github.com/apacha/PrintedMusicSymbolsDataset (The MIT License)
PRINTED_URL = "https://github.com/apacha/OMR-Datasets/releases/download/datasets/PrintedMusicSymbolsDataset.zip"
# @Source: http://sourceforge.net/projects/openomr/ (GNU General Public License, version 2)
OPENOMR_URL = "https://github.com/apacha/OMR-Datasets/releases/download/datasets/OpenOMR-Dataset.zip"
# @Source: https://github.com/Audiveris/omr-dataset-tools (GNU Affero General Public License)
AUDIVERIS_URL = "https://github.com/apacha/OMR-Datasets/releases/download/datasets/AudiverisOmrDataset.zip"

# Get the directory in which to download the dataset
def get_directory_path():
    while True:
        directory_path = input("Enter directory path for dataset downloading: ")
        if not directory_path or not os.path.isdir(directory_path):
            print("Invalid input, try again")
        else:
            while True:
                confirmation = input("The contents of the directory will be deleted. Proceed? (y/n) ").lower()
                if confirmation == 'y':
                    return directory_path
                elif confirmation == 'n':
                    break
                else:
                    continue
        
def create_sub_directory(parent, sub):
    try:
        sub_path = os.path.join(parent, sub)
        os.makedirs(sub_path)
        return sub_path
    except FileExistsError:
        print(f"Error: directory '{sub}' already exists in '{parent}'")
        sys.exit(1)
    except OSError as e:
        print(f"Error: directory '{sub}' could not be created in '{parent}': {e}")
        sys.exit(1)

# Delete the contents of a directory given its path
def delete_dir_contents(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            # If it's a directory, recursively delete its contents
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Error occured: {e}")
            sys.exit(1)

def download_and_extract(out_directory, zipurl):
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as z_file:
            z_file.extractall(out_directory)

# Unify all the class names into a custom standard
def process_dataset_classes(old_path, new_path, ignored, mapping):
    images = os.listdir(old_path)
    for _class in images:
        # Get rid of all ignored classes
        if _class in ignored:
            continue
        src = os.path.join(old_path, _class)
        # If name needs to be mapped, map it
        if _class in mapping:
            fixed_class = mapping[_class]
        else:
            fixed_class = _class
        # Make the directory if it doesn't exist yet
        target = os.path.join(new_path, fixed_class)
        if not os.path.exists(target):
            os.makedirs(target)
        # Copy all files to the processed directory
        src_files = os.listdir(src)
        for file_name in src_files:
            file_path = os.path.join(src, file_name)
            if os.path.isfile(file_path):
                shutil.copy(file_path, target)

def collect_filenames(directory):
    filenames = []
    # Walk through all the subfolders in directory and add name to list
    for(root, dirs, files) in os.walk(directory):
        for file in files:
            filenames.append(file)
    return filenames

if __name__ == "__main__":

    # Get directory
    directory_path = get_directory_path()

    # Delete all contents in the directory
    print("Deleting directory contents... ", end="")
    delete_dir_contents(directory_path)
    print("Done.")

    # Create directory to store unprocessed datasets
    opmr_pure = create_sub_directory(directory_path, "openomr_pure")
    rbl1_pure = create_sub_directory(directory_path, "rebelo1_pure")
    rbl2_pure = create_sub_directory(directory_path, "rebelo2_pure")
    prnt_pure = create_sub_directory(directory_path, "printed_pure")
    audi_pure = create_sub_directory(directory_path, "audivers_pure")

    print("Downloading datasets... ", end="")
    download_and_extract(rbl1_pure, REBELO1_URL)
    download_and_extract(rbl2_pure, REBELO2_URL)
    download_and_extract(prnt_pure, PRINTED_URL)
    download_and_extract(opmr_pure, OPENOMR_URL)
    download_and_extract(audi_pure, AUDIVERIS_URL)
    print("Done.")

    # Extract the symbols from Audiveris using omrdatasettools
    audi_symbols_pure = create_sub_directory(directory_path, "audi_symbols_pure")
    img_generator = AudiverisOmrImageGenerator()
    img_generator.extract_symbols(audi_pure, audi_symbols_pure)

    # Create directory to store processed datasets
    data_proc_path = create_sub_directory(directory_path, "processed_dataset")
    # Store datasets with removed classes and standardized names
    print("Cleaning data... ", end="")
    process_dataset_classes(opmr_pure, data_proc_path, ignored_classes["OpenOMR"], mapped_classes["OpenOMR"])
    process_dataset_classes(rbl1_pure, data_proc_path, ignored_classes["Rebelo1"], mapped_classes["Rebelo1"])
    process_dataset_classes(rbl2_pure, data_proc_path, ignored_classes["Rebelo2"], mapped_classes["Rebelo2"])
    process_dataset_classes(prnt_pure, data_proc_path, ignored_classes["Printed"], mapped_classes["Printed"])
    process_dataset_classes(audi_symbols_pure, data_proc_path, ignored_classes["Audiveris"], mapped_classes["Audiveris"])
    print("Done.")

    # Split into testing, training, and validation subsets
    print("Splitting data... ")
    split_path = os.path.join(directory_path, "split-dataset")
    # Split at a ratio of 80% training, 10% validation, and 10% testing
    splitfolders.ratio(data_proc_path, split_path, seed=2003, ratio=(0.8, 0.1, 0.1))
    print("Done.")

    # Delete original folders
    print("Removing dummy folders... ", end="")
    for root, dirs, files in os.walk(directory_path):
        if "split-dataset" in dirs:
            dirs.remove("split-dataset")
        for dir_name in dirs:
            if dir_name != "split-dataset":
                shutil.rmtree(os.path.join(root, dir_name))
    print("Done.")

    print(f"Dataset is located in {split_path}.")

    # Check how many files in each split
    test_files = collect_filenames(os.path.join(split_path, "test"))
    train_files = collect_filenames(os.path.join(split_path, "train"))
    val_files = collect_filenames(os.path.join(split_path, "val"))

    # Verify the quantity of each set
    total_files = len(test_files) + len(train_files) + len(val_files)
    print("Total files in dataset:", total_files)
    print("Training files compromise", "{:.2%}".format(len(train_files)/total_files), 
        "of the total dataset, with", len(train_files), "images")
    print("Testing files compromise", "{:.2%}".format(len(test_files)/total_files), 
        "of the total dataset, with", len(test_files), "images")
    print("Validation files compromise", "{:.2%}".format(len(val_files)/total_files), 
        "of the total dataset, with", len(val_files), "images")
