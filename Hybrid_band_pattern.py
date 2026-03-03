import cv2
import numpy as np
from termcolor import cprint

from Patterns.Local_Binary_Pattern import calculate_lbp
from Patterns.Local_Ternary_Pattern import calculate_ltp
from Patterns.Local_Directional_Pattern import ldp_process


def hybrid_band_pattern(img):
    cprint("====================================", color="yellow")
    cprint("Extracting Hybrid band pattern.", color="green")
    cprint("====================================", color="yellow")
    # img = cv2.imread("../Dataset/PlantVillage/Pepper__bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG")
    R_band = img[:, :, 0]
    G_band = img[:, :, 1]
    B_band = img[:, :, 2]

    R_local_binary_pattern = calculate_lbp(R_band)
    G_local_binary_pattern = calculate_lbp(G_band)
    B_local_binary_pattern = calculate_lbp(B_band)

    R_local_directional_pattern = ldp_process(R_band)
    G_local_directional_pattern = ldp_process(G_band)
    B_local_directional_pattern = ldp_process(B_band)

    R_local_ternary_pattern = calculate_ltp(R_band)
    G_local_ternary_pattern = calculate_ltp(G_band)
    B_local_ternary_pattern = calculate_ltp(B_band)

    R_local_binary_pattern = cv2.resize(R_local_binary_pattern, (28, 28))
    G_local_binary_pattern = cv2.resize(G_local_binary_pattern, (28, 28))
    B_local_binary_pattern = cv2.resize(B_local_binary_pattern, (28, 28))

    R_local_directional_pattern = cv2.resize(R_local_directional_pattern, (28, 28))
    G_local_directional_pattern = cv2.resize(G_local_directional_pattern, (28, 28))
    B_local_directional_pattern = cv2.resize(B_local_directional_pattern, (28, 28))

    R_local_ternary_pattern = cv2.resize(R_local_ternary_pattern, (28, 28))
    G_local_ternary_pattern = cv2.resize(G_local_ternary_pattern, (28, 28))
    B_local_ternary_pattern = cv2.resize(B_local_ternary_pattern, (28, 28))

    R_local_binary_pattern = np.expand_dims(R_local_binary_pattern, axis=2)
    G_local_binary_pattern = np.expand_dims(G_local_binary_pattern, axis=2)
    B_local_binary_pattern = np.expand_dims(B_local_binary_pattern, axis=2)

    R_local_directional_pattern = np.expand_dims(R_local_directional_pattern, axis=2)
    G_local_directional_pattern = np.expand_dims(G_local_directional_pattern, axis=2)
    B_local_directional_pattern = np.expand_dims(B_local_directional_pattern, axis=2)

    R_local_ternary_pattern = np.expand_dims(R_local_ternary_pattern, axis=2)
    G_local_ternary_pattern = np.expand_dims(G_local_ternary_pattern, axis=2)
    B_local_ternary_pattern = np.expand_dims(B_local_ternary_pattern, axis=2)

    FEAT = np.concatenate(
        [R_local_ternary_pattern, R_local_directional_pattern, R_local_binary_pattern, G_local_ternary_pattern,
         G_local_directional_pattern, G_local_binary_pattern, B_local_ternary_pattern, B_local_directional_pattern,
         B_local_binary_pattern], axis=2)

    histograms = []
    for i in range(FEAT.shape[2]):  # 9 channels
        # Extract the channel (2D array)
        channel = FEAT[:, :, i]
        # Flatten the channel into a 1D array
        channel_flat = channel.flatten()
        # Calculate the histogram using numpy.histogram
        hist, bin_edges = np.histogram(channel_flat, bins=256, range=(0, 256))
        # Append the histogram for this channel to the list
        histograms.append(hist)
    # Final_feat = FEAT.reshape(FEAT.shape[0] * FEAT.shape[1] * FEAT.shape[2])

    return np.array(histograms)


# path = "../Dataset/PlantVillage/Pepper__bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG"
#
# img1 = cv2.imread(path)
# out = hybrid_band_pattern(img1)
# print(out.shape)
