import numpy as np
import cv2
from skimage.feature import local_binary_pattern

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt


def desc_MTP(img, gridHist=(1, 1), t=10, mode=None):
    """
    Applies the Median Ternary Pattern (MTP) descriptor on the image.

    Parameters:
    - img: The input image (grayscale).
    - gridHist: Tuple (numRow, numCol) defining the grid for histogram calculation.
    - t: Threshold value for median comparison.
    - mode: If 'nh', the histograms will be normalized.

    Returns:
    - imgDesc: The descriptor image with the MTP applied (binary pattern applied on the image).
    """
    # Check if the image is grayscale
    if len(img.shape) != 2:
        # raise ValueError("The input image must be a grayscale image.")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.double(img)

    rowNum, colNum = gridHist
    rSize, cSize = img.shape[0] - 2, img.shape[1] - 2

    print(f"Image Shape: {img.shape}")  # Debugging line to check image shape
    link = np.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1]])

    # Extract image intensities based on the link array
    ImgIntensity = np.zeros((rSize * cSize, len(link)))

    for n in range(len(link)):
        corner = link[n, :]
        # Fix the region extraction
        region = img[corner[0] - 1:corner[0] - 1 + rSize, corner[1] - 1:corner[1] - 1 + cSize]
        ImgIntensity[:, n] = region.reshape(-1)

    # Calculate the median
    medianMat = np.median(ImgIntensity, axis=1)

    # Create MTP binary patterns
    Pmtp = (ImgIntensity > (medianMat[:, None] + t))  # Boolean array
    Nmtp = (ImgIntensity < (medianMat[:, None] - t))  # Boolean array

    # Convert boolean arrays into integer values using packbits
    Pmtp_img = np.reshape(np.packbits(Pmtp.astype(np.bool_)), (rSize, cSize))
    Nmtp_img = np.reshape(np.packbits(Nmtp.astype(np.bool_)), (rSize, cSize))

    # Combine Pmtp and Nmtp images as a single descriptor image
    imgDesc = Pmtp_img + 2 * Nmtp_img  # Combining the MTP patterns into a single image for visualization

    imgDesc = cv2.resize(imgDesc, dsize=(28, 28))
    return imgDesc


# # Example usage
# img = cv2.imread('../Dataset/PlantVillage/Pepper__bell___healthy/0a3f2927-4410-46a3-bfda-5f4769a5aaf8___JR_HL 8275.JPG',
#                  cv2.IMREAD_GRAYSCALE)
# imgDesc = desc_MTP(img, gridHist=(2, 2), t=15, mode='nh')
#
# plt.imshow(imgDesc, cmap="gray")
# plt.show()
