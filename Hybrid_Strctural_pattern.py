from keras.applications import ResNet101
from keras.models import Model
import matplotlib.pyplot as plt
import cv2
from termcolor import cprint

from Sub_Functions.Structural_pattern import StructuralPattern
import numpy as np

from Feature_Extraction.Median_Ternary_pattern import desc_MTP


def Hybrid_Structural_Pattern(img):
    cprint("====================================", color="yellow")
    cprint("Extracting Hybrid Structural Patterns.", color="green")
    cprint("====================================", color="yellow")
    SP = StructuralPattern(img)
    # getting the structural pattern
    final_out = SP.get_structural_pattern()
    final_out = cv2.resize(final_out,dsize=(28,28))
    MTP = desc_MTP(img)
    # MTP = cv2.cvtColor(MTP,cv2.COLOR_BGR2GRAY)
    alpha = 0.7
    beta = 0.3
    gamma = 0
    out = cv2.addWeighted(final_out, alpha, MTP, beta, gamma)
    # returning the final output
    out = cv2.resize(out, (28, 28))
    hist, bin_edges = np.histogram(out, bins=256, range=(0, 256))
    # feat = out.reshape(out.shape[0] * final_out.shape[1])  # output shape 224,224
    return np.array(hist)


# img = cv2.imread(
#     '../Dataset/PlantVillage/Pepper__bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert from BGR to RGB for correct colors
#
# final_out = Hybrid_Structural_Pattern(img)
# print(final_out.shape)
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.title('Original Image')
# plt.axis('off')  # Optional: hide axis
# plt.subplot(1, 2, 2)
# plt.imshow(final_out, cmap='gray')
# plt.title('Structural Patterns')
# plt.axis('off')  # Optional: hide axis
# plt.show()
