'''
@breif file to propose symbol regions from input image
'''

import selectivesearch
import numpy as np
import PIL
from PIL import Image as PILImage
import os
import shutil

def nostaff_to_regions(nostaff):
    '''
    returns a set of candidate regions when given an image with staff lines
    removed
    '''
    img_no_alpha = nostaff#[:,:,:3]
    # applying selective search
    img_lbl, regions = selectivesearch.selective_search(
        img_no_alpha,scale=10000, sigma=0, min_size=200)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions that take up the whole damn picture
        if r['size'] > (img_no_alpha.shape[0]-img_no_alpha.shape[0]/10)*(img_no_alpha.shape[1]-img_no_alpha.shape[1]/10):
            continue
        # distorted rects
        x, y, w, h = r['rect']
        #if we just want good squares, we can specify these params
        if w==0 or h==0 or w / h > 10 or h / w > 10:
            continue
        candidates.add(r['rect'])

    #eliminating sub rectangles
    candidates = list(candidates)
    to_remove = set()
    for i in range(len(candidates)):
        x, y, w, h = candidates[i]
        for j in range(len(candidates)):
            if j == i:
                continue  # Skip self-comparison
            x2, y2, w2, h2 = candidates[j]
            # Check if the candidate is within another candidate
            if x2 >= x and y2 >= y and x2 + w2 <= x + w and y2 + h2 <= y + h:
                to_remove.add(candidates[j])

    # Remove the identified sub-rectangles
    for item in to_remove:
        if item in candidates:
            candidates.remove(item)

    return candidates, img_no_alpha

#remove duplicate x and y?

def export_regions(regions, img, folder_path):
    for region in regions:
        x, y, w, h = region
        sub_img = img[y:(y+h),x:(x+w),:]

        # plt.imshow(sub_img, interpolation='nearest')
        # plt.show()

        pil_img = PIL.Image.fromarray(sub_img)
        img_name = f"{x}_{y+h}_{w}_{h}"
        # print(img_name)
        # print(sub_img.shape)

        full_path = os.path.join(folder_path, img_name) + ".png"

        pil_img.save(full_path)

input_img_path = 'new_image.png'

# Load the image using PIL
image = PILImage.open(input_img_path)
image_rgba = image.convert('RGB')
# Convert the PIL.Image.Image to a numpy array
image_np = np.array(image_rgba)

regions, img = nostaff_to_regions(image_np)

output_path = "proposal_output"
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)
export_regions(regions,img,output_path)