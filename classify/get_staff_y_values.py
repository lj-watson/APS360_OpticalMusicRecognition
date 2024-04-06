'''
@breif file to get the locations of staff lines
'''

from PIL import Image, ImageDraw
import json

# Load the image
img = Image.open("../release/input.png")
width, height = img.size

# Convert the image to grayscale then to black and white based on the threshold
img_bw = img.convert('1')  # Convert to grayscale

# To store the y-coordinates of the rows that are more than half black pixels
staff_lines = []

# Iterate over each row
for y in range(height):
    black_pixels = 0
    # Iterate over each pixel in the row
    for x in range(width):
        pixel = img_bw.getpixel((x, y))
        if pixel == 0:  # Check if the pixel is black
            black_pixels += 1
    # Check if more than half of the pixels in the row are black
    if black_pixels > (width*2 / 3):
        staff_lines.append(y)

staff_lines.sort()
    
# Initialize the result array with the first element
staff_lines_cleaned = [staff_lines[0]]

first_of_group = staff_lines[0]
    
# Iterate through the sorted array starting from the second element
for i in range(1, len(staff_lines)):
    # If the current element is outside the range from the first of the current group
    if staff_lines[i] - first_of_group > 4:
        # Start a new group with this element
        first_of_group = staff_lines[i]
        staff_lines_cleaned.append(staff_lines[i])

draw = ImageDraw.Draw(img)
    
# Highlight color and line width
highlight_color = "red"
line_width = 2

# Draw a line across the width of the image at each y-value
for staff in staff_lines_cleaned:
    draw.line([(0, staff), (width, staff)], fill=highlight_color, width=line_width)

# Save the modified image
img.save("staff_lines_location.png")
        
# Write staff coords to json file
filename = 'octave_y_values.json'
output_path = "../audio"
json_path = f'{output_path}/{filename}'
with open(json_path, 'w') as file:
    json.dump(staff_lines_cleaned, file)
