import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

img = cv2.imread(
    "Dataset/PlantVillage/Pepper__bell___healthy/0b76f650-27cf-4b62-b3ad-c97d81e0db0c___JR_HL 8554.JPG")

median_filter = cv2.medianBlur(img, ksize=3)
gaussian_filter = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0.6, sigmaY=0.3)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Split LAB planes - cv2.split() returns TUPLE, so convert to list
lab_planes = list(cv2.split(lab))  # ✅ Convert tuple → list

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

# Apply CLAHE to L channel only (index 0)
lab_planes[0] = clahe.apply(lab_planes[0])

# Merge LAB planes back
clahe_img = cv2.merge(lab_planes)

# Convert back to BGR
clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)

plt.subplot(2, 2, 1)
plt.title("Gaussian Filter")
plt.imshow(gaussian_filter)
plt.subplot(2, 2, 2)
plt.title("Median Filter")
plt.imshow(median_filter)
plt.subplot(2, 2, 3)
plt.title("Clahe Filter")
plt.imshow(clahe_bgr)
plt.subplot(2, 2, 4)
plt.title("Original Image")
plt.imshow(img)
os.makedirs(f"Samples/",exist_ok=True)
plt.savefig(f"Samples/sample2.jpg", dpi=900)
plt.show()
