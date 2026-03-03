import cv2
from matplotlib import pyplot as plt
from vit_keras import vit, utils
from termcolor import cprint


def transformer_based_feature(img):
    cprint("====================================", color="yellow")
    cprint("Extracting Transformer Based Feature.", color="green")
    cprint("====================================", color="yellow")
    img_size = 256

    model = vit.vit_b16(
        image_size=img_size,
        activation='sigmoid',
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )

    # url = "../Dataset/PlantVillage/Pepper__bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG"
    # image = cv2.imread(url)
    X = vit.preprocess_inputs(img).reshape(1, img_size, img_size, 3)
    y = model.predict(X)

    return y


# img = cv2.imread(
#     '../Dataset/PlantVillage-Dataset/raw/color/Peach___Bacterial_spot/f6dde179-db72-40f1-beea-3566d1ca4eed___Rutg._Bact.S 1291.JPG')
# roi = transformer_based_feature(img)
# print(roi.shape)
# plt.subplot(1, 2, 1)
# img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(roi)
# plt.show()

# link  https://github.com/faustomorales/vit-keras
