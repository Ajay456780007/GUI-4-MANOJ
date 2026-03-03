import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from rembg import remove
from termcolor import cprint


def ROI_Extraction(img_, r=False):
    cprint("====================================", color="yellow")
    cprint("Extracting ROI.", color="green")
    cprint("====================================", color="yellow")
    # img_ = cv2.resize(img_, (4, 400))
    # --------- region of Interest Segmentation ---------------
    # img = cv2.resize(img_, (224, 224))
    hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)  # convert to hue saturation value
    # range of green color in HSV
    l_g = np.array([30, 120, 50])  # lower bound
    u_g = np.array([120, 255, 100])  # upper bound
    # mask
    mask = cv2.inRange(hsv, l_g, u_g)
    # Apply morphological operations to remove small noise and fill gaps in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Adjust kernel size
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # If contours are found, proceed to extract the ROI
    if r:
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        # Get bounding box coordinates of the largest contour
        # x, y, w, h = cv2.boundingRect(contour)
        x, y, w, h = cv2.boundingRect(contour)
        # Extract the region of interest (ROI)
        roi = img_[x: x + w, y: y + h]
        if w == 0 or h == 0 or x == 0 or y == 0:
            new_img_rgba = img_
        else:
            new_img_rgba = cv2.resize(roi, (224, 224))
    else:
        out = remove(img_)
        new_img_rgba = cv2.cvtColor(out, cv2.COLOR_BGRA2RGB)
    return new_img_rgba


# img = cv2.imread(
#     '../Dataset/PlantVillage-Dataset/raw/color/Peach___Bacterial_spot/f6dde179-db72-40f1-beea-3566d1ca4eed___Rutg._Bact.S 1291.JPG')
# roi = ROI_Extraction(img)
# plt.subplot(1, 2, 1)
# img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(roi)
# plt.show()