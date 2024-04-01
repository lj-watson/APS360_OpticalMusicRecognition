'''
@brief Get the octaves for each symbol based on the given y value
'''

import json

index_mapping_gclef = {
    0: 'C4',
    1: 'D4',
    2: 'E4',
    3: 'F4',
    4: 'G4',
    5: 'A4',
    6: 'B4',
    7: 'C5',
    8: 'D5',
    9: 'E5',
    10: 'F5',
    11: 'G5'
}

index_mapping_cclef = {
    0: 'E2',
    1: 'F2',
    2: 'G2',
    3: 'A2',
    4: 'B2',
    5: 'C3',
    6: 'D3',
    7: 'E3',
    8: 'F3',
    9: 'G3',
    10: 'A3',
    11: 'B3'
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

# Fill array of all octave locations
octave_values = []
bottom = octave_data[0]
octave_values.append(bottom)
octave_values.append(bottom - average_distance)
octave_values.append(bottom - octave_data*2)
for i in range(9):
    octave_values.append(bottom+average_distance*(i+1))

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

symbol_octaves = []
for pos in symbol_y_data:
    closest_index = min(range(len(octave_values)), key=lambda x: abs(octave_values[x]-pos))
    if clef == 'G-Clef':
        symbol_octaves.append(index_mapping_gclef[closest_index])
    else:
        symbol_octaves.append(index_mapping_cclef[closest_index])

# Write symbol octaves to json file
text_string = ':'.join(symbol_octaves)
text_string = f'<{text_string}>'
data = {"octaves": symbol_octaves}
filename = 'symbols.json'
with open(filename, 'w') as file:
    json.dump(data, file)