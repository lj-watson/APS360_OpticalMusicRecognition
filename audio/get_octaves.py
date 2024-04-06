'''
@brief Get the octaves for each symbol based on the given y value
'''

import json
from PIL import Image, ImageDraw
import os
import numpy as np

# Octaves are counted from the top downwards
index_mapping_gclef = {
    0: 'G5',
    1: 'F5',
    2: 'E5',
    3: 'D5',
    4: 'C5',
    5: 'B4',
    6: 'A4',
    7: 'G4',
    8: 'F4',
    9: 'E4',
    10: 'D4',
    11: 'C4'
 }

index_mapping_cclef = {
    0: 'B4',
    1: 'A4',
    2: 'G3',
    3: 'F3',
    4: 'E3',
    5: 'D3',
    6: 'C3',
    7: 'B2',
    8: 'A2',
    9: 'G2',
    10: 'F2',
    11: 'E2'
}

# Get octave y values from input image
with open("octave_y_values.json", 'r') as file:
    octave_data = json.load(file)
# Get average distance between each octave
total_distance = 0
for i in range(1, len(octave_data)):
    distance = abs(octave_data[i] - octave_data[i-1])  # Use absolute value to ensure distance is positive
    total_distance += distance
average_distance = total_distance / (len(octave_data) - 1)
# Since this is the average distance between each line, divide by 2
average_distance = average_distance / 2

# Fill array of all octave locations
octave_values = []
top = octave_data[0]
octave_values.append(top - average_distance)
for i in range(11):
    octave_values.append(top+(average_distance*i))

# Load the image
img = Image.open("../release/input.png")
draw = ImageDraw.Draw(img)
    
# Highlight color and line width
highlight_color = "red"
line_width = 2

# Draw a line across the width of the image at each y-value
for staff in octave_values:
    draw.line([(0, staff), (img.width, staff)], fill=highlight_color, width=line_width)

# Save the modified image
img.save("staff_lines_location_updated.png")

# Find out if we are in treble clef or base clef
with open("symbols.json", 'r') as file:
    symbol_data = json.load(file)

symbols = symbol_data['classification']
symbol_items = symbols.strip("<>").split(":")

clef = 'G-Clef'
for _, symbol in enumerate(symbol_items):
    if "Clef" in symbol:
        clef = symbol
        break

# Loop through each symbol y location and classify it into an octave
with open("symbol_y_values.json", 'r') as file:
    symbol_y_data = json.load(file)

# Make a list of notes that are upside down
potential_upsidedown_list = []
for index, symbol in enumerate(symbol_items):
    if (symbol == 'Quarter-Note' or symbol == 'Eight-Note' or 
    symbol == 'Half-Note' or symbol == 'Sixteenth-Note' or 
    symbol == 'Sixty-Four-Note' or symbol == 'Thirty-Two-Note'):
        potential_upsidedown_list.append(index)

upsidedown_list = []
imgs_path = '../classify/proposal_output'
all_files = os.listdir(imgs_path) 
all_files.sort()
for index in potential_upsidedown_list:

    filepath = os.path.join(imgs_path, all_files[index])
    
    image = Image.open(filepath)
    # Convert the image to grayscale (optional, depends on the image)
    gray_image = image.convert('L')
    
    # Get the dimensions of the image
    width, height = gray_image.size
    
    # Split the image into top and bottom halves
    top_half = gray_image.crop((0, 0, width, height // 2))
    bottom_half = gray_image.crop((0, height // 2, width, height))
    
    # Count black pixels in each half
    # In a grayscale image, black is 0 and white is 255
    top_black_pixels = sum(pixel == 0 for pixel in top_half.getdata())
    bottom_black_pixels = sum(pixel == 0 for pixel in bottom_half.getdata())
    
    # Note is upsidedown
    if top_black_pixels > bottom_black_pixels:
        upsidedown_list.append(index)

with open("symbol_h_values.json", 'r') as file:
    symbol_h_data = json.load(file)

symbol_octaves = []
for index, pos in enumerate(symbol_y_data):
    # Need to check for upside down note
    if index in upsidedown_list:
        pos = pos - symbol_h_data[index]
        closest_index = min(range(len(octave_values)), key=lambda x: abs(octave_values[x]-pos))
        if clef == 'G-Clef':
            if pos > max(octave_values) or pos < min(octave_values):
                symbol_octaves.append(index_mapping_gclef[closest_index])
            else:
                symbol_octaves.append(index_mapping_gclef[closest_index+1])
        else:
            if pos > max(octave_values) or pos < min(octave_values):
                symbol_octaves.append(index_mapping_cclef[closest_index])
            else:
                symbol_octaves.append(index_mapping_cclef[closest_index+1])
    else:
        closest_index = min(range(len(octave_values)), key=lambda x: abs(octave_values[x]-pos))
        if clef == 'G-Clef':
            if  pos > max(octave_values) or pos < min(octave_values):
                symbol_octaves.append(index_mapping_gclef[closest_index])
            else:
                symbol_octaves.append(index_mapping_gclef[closest_index-1])
        else:
            if  pos > max(octave_values) or pos < min(octave_values):
                symbol_octaves.append(index_mapping_cclef[closest_index])
            else:
                symbol_octaves.append(index_mapping_cclef[closest_index-1])

# Write symbol octaves to json file
text_string = ':'.join(symbol_octaves)
text_string = f'<{text_string}>'
data = {"octaves": text_string}
filename = 'octaves.json'
with open(filename, 'w') as file:
    json.dump(data, file)