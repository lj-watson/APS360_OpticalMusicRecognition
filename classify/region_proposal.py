'''
@breif file to propose symbol regions from input image
'''

import selectivesearch
import sys
import numpy as np
import PIL
from PIL import Image as PILImage
import os

def nostaff_to_regions(nostaff):
    '''
    returns a set of candidate regions when given an image with staff lines
    removed
    '''
    img_no_alpha = nostaff[:,:,:3]
    # applying selective search
    img_lbl, regions = selectivesearch.selective_search(
        img_no_alpha,scale=100000, sigma=0, min_size=2000)

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

    # # draw rectangles on the original image
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    # ax.imshow(img_no_alpha)
    # for x, y, w, h in candidates:
    #     #print(x, y, w, h)
    #     rect = mpatches.Rectangle(
    #         (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    #     ax.add_patch(rect)
    # plt.show()
    return candidates, img_no_alpha

    #remove duplicate x and y?

def export_regions(regions, img, folder_path):
    for region in regions:
        x, y, w, h = region
        sub_img = img[y:(y+h),x:(x+w),:]

        # plt.imshow(sub_img, interpolation='nearest')
        # plt.show()

        pil_img = PIL.Image.fromarray(sub_img)
        img_name = f"x{x}_y{y}_w{w}_h{h}"
        # print(img_name)
        # print(sub_img.shape)

        full_path = os.path.join(folder_path, img_name)

        pil_img.save(full_path, "PNG")

input_img_path = 'new_image.png'

# Load the image using PIL
image = PILImage.open(input_img_path)
image_rgba = image.convert('RGBA')
# Convert the PIL.Image.Image to a numpy array
image_np = np.array(image_rgba)

regions, img = nostaff_to_regions(image_np)

output_path = "proposal_output"
os.makedirs(output_path, exist_ok=True)
export_regions(regions,img,output_path)