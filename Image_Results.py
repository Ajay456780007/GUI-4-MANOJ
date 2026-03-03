import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    features = cv2.resize(features, (250, 250))
    # feat = features.reshape(features.shape[0] * features.shape[1] * features.shape[2])
    return features


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

    R_local_binary_pattern = cv2.resize(R_local_binary_pattern, (250, 250))
    G_local_binary_pattern = cv2.resize(G_local_binary_pattern, (250, 250))
    B_local_binary_pattern = cv2.resize(B_local_binary_pattern, (250, 250))

    R_local_directional_pattern = cv2.resize(R_local_directional_pattern, (250, 250))
    G_local_directional_pattern = cv2.resize(G_local_directional_pattern, (250, 250))
    B_local_directional_pattern = cv2.resize(B_local_directional_pattern, (250, 250))

    R_local_ternary_pattern = cv2.resize(R_local_ternary_pattern, (250, 250))
    G_local_ternary_pattern = cv2.resize(G_local_ternary_pattern, (250, 250))
    B_local_ternary_pattern = cv2.resize(B_local_ternary_pattern, (250, 250))

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

    # Final_feat = FEAT.reshape(FEAT.shape[0] * FEAT.shape[1] * FEAT.shape[2])

    return FEAT


from keras.applications import ResNet101
from keras.models import Model
import matplotlib.pyplot as plt
import cv2
from termcolor import cprint

from Sub_Functions.Structural_pattern import StructuralPattern
import numpy as np


def Hybrid_Structural_Pattern(img):
    cprint("====================================", color="yellow")
    cprint("Extracting Hybrid Structural Patterns.", color="green")
    cprint("====================================", color="yellow")
    SP = StructuralPattern(img)
    # getting the structural pattern
    final_out = SP.get_structural_pattern()
    # returning the final output
    final_out = cv2.resize(final_out, (250, 250))
    # feat = final_out.reshape(final_out.shape[0] * final_out.shape[1])  # output shape 224,224
    return final_out


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
            cropped_img = img_
        else:
            cropped_img = cv2.resize(roi, (224, 224))
    else:
        out = remove(img_)
        new_img_rgba = cv2.cvtColor(out, cv2.COLOR_BGRA2RGB)

        gray = cv2.cvtColor(new_img_rgba, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        cropped_img = img_[y:y + h, x:x + w]
        cropped_img = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)
    return cropped_img


import tensorflow as tf
import numpy as np
from vit_keras import vit, utils
import cv2
import matplotlib.pyplot as plt


def transformer_based_feature(img):
    img_size = 256

    model = vit.vit_b16(
        image_size=img_size,
        activation='sigmoid',
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )

    target_layer = model.get_layer('Transformer/encoderblock_6')

    intermediate_model = tf.keras.Model(
        inputs=model.input,
        outputs=target_layer.output
    )

    X = vit.preprocess_inputs(img).reshape(1, img_size, img_size, 3)

    # 'prediction' is likely a tuple: (hidden_states, attention_weights)
    prediction = intermediate_model.predict(X)

    # 1. Check if it's a tuple and grab the first part (the features)
    if isinstance(prediction, (list, tuple)):
        tokens = prediction[0]
    else:
        tokens = prediction

    # 2. Slice and Reshape
    # tokens shape is (1, 257, 768)
    spatial_tokens = tokens[0, 1:, :]  # Remove Batch and CLS token
    feature_map = spatial_tokens.reshape(16, 16, 768)

    return feature_map


base_path1 = "Dataset/PlantVillage"

folders1 = os.listdir(base_path1)
full_folder_path1 = [os.path.join(base_path1, i) for i in folders1]

for index, im in enumerate(full_folder_path1):
    images = os.listdir(im)[:2]

    images_path = [os.path.join(full_folder_path1[index], m) for m in images]

    # for index2, img in enumerate(images_path):
    #     img_loaded = cv2.imread(img)
    #     Roi_Ex = ROI_Extraction(img_loaded)
    #     plt.imshow(Roi_Ex)
    #     os.makedirs(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/", exist_ok=True)
    #     plt.imsave(f"Image_Results/DB/{folders1[index]}/Sample1/Roi_Extraction.jpg", dpi=600)
    #     plt.close()
    #
    #     glcm_feat = glcm_statistical_features(img_loaded)
    #     os.makedirs(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/GLCM_Feat/", exist_ok=True)
    #     plt.imshow(glcm_feat[:,:,0])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/GLCM_Feat/Energy.jpg",dpi=800)
    #     plt.close()
    #     plt.imshow(glcm_feat[:, :, 1])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/GLCM_Feat/disimilarity.jpg", dpi=800)
    #     plt.close()
    #     plt.imshow(glcm_feat[:, :, 2])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/GLCM_Feat/Homogenity.jpg", dpi=800)
    #     plt.close()
    #     plt.imshow(glcm_feat[:, :, 3])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/GLCM_Feat/Entropy.jpg", dpi=800)
    #     plt.close()
    #     plt.imshow(glcm_feat[:, :, 4])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/GLCM_Feat/Contrast.jpg", dpi=800)
    #     plt.close()
    #
    #     HBP = hybrid_band_pattern(img_loaded)
    #     os.makedirs(f"Image_Results/DB/{folders1[index]}/Sample{index2+1}/Hybrid_band_pattern/",exist_ok=True)
    #     plt.imshow(HBP[:,:,0])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2+1}/Hybrid_band_pattern/R_band_lbp.jpg",dpi=900)
    #     plt.close()
    #     plt.imshow(HBP[:, :, 1])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/Hybrid_band_pattern/R_band_ldp.jpg", dpi=900)
    #     plt.close()
    #     plt.imshow(HBP[:, :, 2])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/Hybrid_band_pattern/R_band_ltp", dpi=900)
    #     plt.close()
    #     plt.imshow(HBP[:, :, 3])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/Hybrid_band_pattern/G_band_lbp.jpg", dpi=900)
    #     plt.close()
    #     plt.imshow(HBP[:, :, 4])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/Hybrid_band_pattern/G_band_ldp.jpg", dpi=900)
    #     plt.close()
    #     plt.imshow(HBP[:, :, 5])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/Hybrid_band_pattern/G_band_ltp.jpg", dpi=900)
    #     plt.close()
    #     plt.imshow(HBP[:, :, 6])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/Hybrid_band_pattern/B_band_lbp.jpg", dpi=900)
    #     plt.close()
    #     plt.imshow(HBP[:, :, 7])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/Hybrid_band_pattern/B_band_ldp.jpg", dpi=900)
    #     plt.close()
    #     plt.imshow(HBP[:, :, 8])
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}/Hybrid_band_pattern/B_band_ltp.jpg", dpi=900)
    #     plt.close()
    #
    #     HSP = Hybrid_Structural_Pattern(img_loaded)
    #     plt.imshow(HSP)
    #     os.makedirs(f"Image_Results/DB/{folders1[index]}/Sample{index2 +1}/",exist_ok=True)
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 +1}/Hybrid_Structural_pattern.jpg",dpi=600)
    #     plt.close()
    #
    #     trans_feat=transformer_based_feature(img)
    #     plt.imshow(trans_feat[:,:,1])
    #     os.makedirs(f"Image_Results/DB/{folders1[index]}/Sample{index2 +1}/",exist_ok=True)
    #     plt.savefig(f"Image_Results/DB/{folders1[index]}/Sample{index2 +1}/transformer_feat.jpg",dpi=900)
    #     plt.close()

    for index2, img_path in enumerate(images_path):

        img_loaded = cv2.imread(img_path)

        # -------- Base directory for this sample --------
        base_dir = f"Image_Results/DB/{folders1[index]}/Sample{index2 + 1}"
        os.makedirs(base_dir, exist_ok=True)

        class_name = folders1[index]

        # ================= SAVE ORIGINAL IMAGE =================
        cv2.imwrite(f"{base_dir}/Original_Image.jpg", img_loaded)

        output_img = img_loaded.copy()
        h, w, _ = output_img.shape
        class_name = folders1[index]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        line_spacing = 4
        top_bottom_padding = 8
        side_margin = int(0.05 * w)

        # ---------- Word wrapping ----------
        words = class_name.split("_")
        lines = []
        current = ""

        for word in words:
            test = current + "_" + word if current else word
            (tw, th), _ = cv2.getTextSize(test, font, font_scale, thickness)

            if tw <= w - 2 * side_margin:
                current = test
            else:
                lines.append(current)
                current = word

        lines.append(current)

        # ---------- Tight banner height ----------
        text_height = len(lines) * (th + line_spacing) - line_spacing
        banner_height = text_height + 2 * top_bottom_padding

        # ---------- Draw banner ----------
        cv2.rectangle(output_img, (0, 0), (w, banner_height), (0, 0, 0), -1)

        # ---------- Draw text ----------
        y = top_bottom_padding + th

        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            x = (w - tw) // 2

            cv2.putText(output_img, line, (x, y),
                        font, font_scale, (0, 0, 0),
                        thickness + 2, cv2.LINE_8)

            cv2.putText(output_img, line, (x, y),
                        font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_8)

            y += th + line_spacing

        cv2.imwrite(f"{base_dir}/Output.png", output_img)

        # ================= ROI EXTRACTION =================
        Roi_Ex = ROI_Extraction(img_loaded)
        plt.imshow(Roi_Ex)
        plt.axis("off")
        plt.savefig(f"{base_dir}/Roi_Extraction.jpg", dpi=600, bbox_inches='tight')
        plt.close()

        # ================= GLCM FEATURES =================
        glcm_feat = glcm_statistical_features(img_loaded)
        glcm_dir = f"{base_dir}/GLCM_Feat"
        os.makedirs(glcm_dir, exist_ok=True)

        glcm_names = ["Energy", "Dissimilarity", "Homogeneity", "Entropy", "Contrast"]

        for i, name in enumerate(glcm_names):
            plt.imshow(glcm_feat[:, :, i], cmap="gray")
            plt.axis("off")
            plt.savefig(f"{glcm_dir}/{name}.jpg", dpi=800, bbox_inches='tight')
            plt.close()

        # ================= HYBRID BAND PATTERN =================
        HBP = hybrid_band_pattern(img_loaded)
        hbp_dir = f"{base_dir}/Hybrid_band_pattern"
        os.makedirs(hbp_dir, exist_ok=True)

        hbp_names = [
            "R_band_lbp", "R_band_ldp", "R_band_ltp",
            "G_band_lbp", "G_band_ldp", "G_band_ltp",
            "B_band_lbp", "B_band_ldp", "B_band_ltp"
        ]

        for i, name in enumerate(hbp_names):
            plt.imshow(HBP[:, :, i], cmap="gray")
            plt.axis("off")
            plt.savefig(f"{hbp_dir}/{name}.jpg", dpi=900, bbox_inches='tight')
            plt.close()

        # ================= HYBRID STRUCTURAL PATTERN =================
        HSP = Hybrid_Structural_Pattern(img_loaded)
        plt.imshow(HSP, cmap="gray")
        plt.axis("off")
        plt.savefig(f"{base_dir}/Hybrid_Structural_pattern.jpg", dpi=600, bbox_inches='tight')
        plt.close()

        # ================= TRANSFORMER FEATURE =================
        trans_feat = transformer_based_feature(img_loaded)
        plt.imshow(trans_feat[:, :, 1])
        plt.axis("off")
        plt.savefig(f"{base_dir}/Transformer_Feature.jpg", dpi=900, bbox_inches='tight')
        plt.close()
