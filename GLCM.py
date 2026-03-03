import cv2
import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint

from Sub_Functions.GLCM_Feature import GLCM_Features


def glcm_statistical_features(image):
    cprint('')
    cprint("[⚠️] Getting GLCM features ", color='grey', on_color='on_yellow')
    cprint("================================", color='blue')
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image, (120, 120))
    GF = GLCM_Features()

    energy = GF.fast_glcm_ASM(image)[0]
    # plt.imshow(energy)
    # plt.axis('off')
    # plt.savefig("image\\d1\\one\\glcm\\glcm_ASM.png", bbox_inches='tight', pad_inches=0)
    # plt.show(block=False)

    dissimilarity = GF.fast_glcm_dissimilarity(image)
    # plt.imshow(dissimilarity)
    # plt.axis('off')
    # plt.savefig("image\\d1\\one\\glcm\\glcm_dissimilarity.png", bbox_inches='tight', pad_inches=0)
    # plt.show(block=False)

    homogeneity = GF.fast_glcm_homogeneity(image)
    # plt.imshow(homogeneity)
    # plt.axis('off')
    # plt.savefig("image\\d1\\one\\glcm\\glcm_homogeneity.png", bbox_inches='tight', pad_inches=0)
    # plt.show(block=False)

    entropy = GF.fast_glcm_entropy(image)
    # plt.imshow(entropy)
    # plt.axis('off')
    # plt.savefig("image\\d1\\one\\glcm\\glcm_entropy.png", bbox_inches='tight', pad_inches=0)
    # plt.show(block=False)

    contrast = GF.fast_glcm_contrast(image)
    # plt.imshow(contrast)
    # plt.axis('off')
    # plt.savefig("image\\d1\\one\\glcm\\glcm_contrast.png", bbox_inches='tight', pad_inches=0)
    # plt.show(block=False)

    features = np.zeros(shape=(image.shape[0], image.shape[1], 5))
    features[:, :, 0] = energy
    features[:, :, 1] = dissimilarity
    features[:, :, 2] = homogeneity
    features[:, :, 3] = entropy
    features[:, :, 4] = contrast
    features = np.nan_to_num(features)
    features = cv2.resize(features, (28, 28))
    # feat = features.reshape(features.shape[0] * features.shape[1] * features.shape[2])
    histograms =[]

    for i in range(features.shape[2]):
        channel = features[:, :, i]
        channel_flat = channel.flatten()
        hist, bin_edges = np.histogram(channel_flat, bins=256, range=(0, 256))
        histograms.append(hist)
    return np.array(histograms)

# img = cv2.imread("../Dataset/PlantVillage/Pepper__bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG")
# print(img.shape)
# features = glcm_statistical_features(img)
# print(features.shape)
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(features[:, :, 4],cmap="gray")
# plt.show()
