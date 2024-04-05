"""
@brief Given image folder with their location "x_y.png" in the name, rearrange them
from left to right
"""

import os
import json
import sys

def get_position_from_filename(filename):
    # Returns the (x, y) position from the given file
    base = os.path.basename(filename)  # Extracts filename from path
    name, ext = os.path.splitext(base)  # Separates the extension
    if "_" in name and ext == ".png":
        try:
            x, y, w, h = map(int, name.split("_"))
            return x, y, h
        except ValueError:
            pass
    return None

# Get the image directory path
img_dir = "proposal_output"
if not img_dir or not os.path.isdir(img_dir):
        print("Could not find region proposal output directory")
        sys.exit(1)

# Extract positions
positions = []
for filename in os.listdir(img_dir):
    pos = get_position_from_filename(filename)
    if pos:
        positions.append((pos, filename))

# Sort images based on the x position
positions.sort()

# Rearrange and rename files in the directory
y_values = []
h_values = []
for i, ((x, y, h), filename) in enumerate(positions, 1):
    new_name = f"{i:03d}.png"
    old_path = os.path.join(img_dir, filename)
    new_path = os.path.join(img_dir, new_name)
    os.rename(old_path, new_path)
    y_values.append(y)
    h_values.append(h)

# Write y values to json file
filename = 'symbol_y_values.json'
output_path = "../audio"
json_path = f'{output_path}/{filename}'
with open(json_path, 'w') as file:
    json.dump(y_values, file)
filename = 'symbol_h_values.json'
json_path = f'{output_path}/{filename}'
with open(json_path, 'w') as file:
    json.dump(h_values, file)