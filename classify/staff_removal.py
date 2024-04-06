'''
@breif file to remove staff lines from music score
'''
import cv2
import numpy as np
import os
import sys
import json
from PIL import Image as PILImage

def staffline_removal_and_coords(img, padding = 500):
    # converts the image to a gray-scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply binary inverse thresholding with Otsu's method, first index is threshold value, and second value is binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # creating kernel to run through the entire image pixel by p ixel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # converts the image to a full black and white image
    img_bw = cv2.bitwise_not(cleaned)
    # apply binary inverse thresholding with Otsu's method, first index is threshold value, and second value is binary image
    thresh_2 = cv2.threshold(img_bw, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    rows, cols = thresh_2.shape     # separates img_bw into rows and columns
    staff_lines = []                # creates a list of the staff lines in the image (length should not be larger than 5)
    line_thickness_values = []      # stores the thickness of the lines so we can take max, min, or average afterwards
    horizontal_spans = []           # store the lengths of horizontal segments

    # Detect vertical segments representing staff lines
    for col in range(cols):
        start = None
        for row in range(rows):
          # this detects that when we hit the start of a black line and then a white line, it indicates a staff line
            if thresh_2[row, col] == 255 and start is None:
                start = row
            elif thresh_2[row, col] == 0 and start is not None:
                end = row
                length = end - start          # computes length of the staff line
                if length > 0:                # assuming staff lines are vertical and at least 1 pixel thick
                  if len(staff_lines) >= 5:   # maximum of 5 staff lines will be analyzed
                    break
                  else:
                    staff_lines.append((start, col, end))       # start and end are the top and bottom y values of line
                    line_thickness_values.append(int(length))   # line thickness values put into a list
                start = None

    # creates a list that takes the average of the top and bottom y values of each staff line
    staff_line_coords = []
    for line in staff_lines:
        staff_line_coords.append((line[0] + line[2]) / 2)

    # converts staff_line_coords to a numpy array
    staff_line_coords_np = np.array(staff_line_coords)

    # Scan horizontally to determine the maximum horizontal span of staff lines
    for row in range(rows):
        start = None
        for col in range(cols):
            if thresh_2[row, col] == 255 and start is None:
                start = col
            elif thresh_2[row, col] == 0 and start is not None:
                end = col
                length = end - start
                horizontal_spans.append(length)
                start = None  # Reset for potentially new line detection

    # Calculate the maximum length of horizontal segments for staff lines
    if horizontal_spans:
        max_length = int(max(horizontal_spans))
    else:
        # this is an arbitrary value
        max_length = 20

    # ensures that images with a line thickness anywhere in the list of 1 will take 1 as its value for line thickness
    if line_thickness_values:
        if 1 in line_thickness_values:
            line_thickness = 1
        else:
            line_thickness = int(np.max(line_thickness_values))
    else:
        line_thickness = 2

    # Morphological operation to detect horizontal lines in img_bw (staff lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max_length, line_thickness))
    detected_lines = cv2.morphologyEx(thresh_2, cv2.MORPH_OPEN, horizontal_kernel, iterations = 1)

    # Find contours of the detected lines
    contours = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #print(len(contours))

    # Initialize variables to store the bounding box coordinates
    top_y = float('inf')
    bottom_y = 0
    left_x = float('inf')
    right_x = 0

    # Loop over the contours to find the bounding box of the staff lines
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)    # x-coordinate, y-coordinate, width, height
        top_y = min(top_y, y)                     # Need the minimum value between inf and the highest y value
        bottom_y = max(bottom_y, y + h)           # Need the maximum value between 0 and the lowest y value
        left_x = min(left_x, x)                   # Need the minimum value between infinity and the highest x value
        right_x = max(right_x, x + w)             # Need the maximum value between 0 and the lowest x value
        staff_line_coords.append((top_y+bottom_y)/2)

    # Add padding to the top and bottom coordinates of the bounding box to ensure that notes don't get cropped out too
    top_y = max(top_y - padding, 0)
    bottom_y = min(bottom_y + padding, img_bw.shape[0])

    # Crop the image to the bounding box with padding
    cropped_image = img_bw[top_y:bottom_y, left_x:right_x]
    # apply binary inverse thresholding with Otsu's method, first index is threshold value, and second value is binary image
    thresh_3 = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    if line_thickness == 1:
      # checks if contours has length 3 which would return (1)image, (2)contours, (3)hierarchy
      if len(contours) == 3:
        contours = contours[1]
      else:
        contours = contours[0]

      # draws white lines over detected horizontal lines
      for contour in contours:
        removed = cv2.drawContours(cropped_image, [contour], -1, (255, 255, 255), line_thickness)

      # Create a kernel for the repair operation, which is larger than the one used to detect the staff lines
      # This kernel should be wide enough to connect components of the notes but not so tall as to recreate the staff lines
      repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_thickness, 4))
      # Apply a close morphological operation to repair the image
      final_result = cv2.morphologyEx(removed, cv2.MORPH_CLOSE, repair_kernel, iterations = line_thickness)

      return final_result, staff_line_coords
      #cv2_imshow(img_bw)
      #print(line_thickness)

      #return result
      #cv2_imshow(result)

    elif line_thickness > 1:
      # Set the line thickness to at least 2 for the morphological operations to take effect
      line_thickness = max(2, int(np.max(line_thickness_values)))
      # Create kernels for removing and repairing staff lines
      vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_thickness))
      horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_thickness, 1))

      # Remove staff lines
      remove_lines = cv2.morphologyEx(thresh_3, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

      # Repair notes
      repaired_img = cv2.morphologyEx(remove_lines, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
      final_result = cv2.bitwise_not(repaired_img)

      return final_result, staff_line_coords

input_img_path = "../release/input.png"
if not os.path.isfile(input_img_path):
    print(f"The path {input_img_path} does not exist or is not a file. Please make sure the image is called 'input.png'.")
    sys.exit(1)

# Load the image using PIL
image = PILImage.open(input_img_path)
# Convert the PIL.Image.Image to a numpy array
image_np = np.array(image)

res, staff_coords = staffline_removal_and_coords(image_np)

# Save new image
cv2.imwrite('new_image.png', res)