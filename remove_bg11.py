import matplotlib.pyplot as plt
from rembg import remove
import cv2

img = cv2.imread(
    "../Dataset/PlantVillage/Potato___Early_blight/1d466431-007d-4b3b-bd45-b09f1a6f7bad___RS_Early.B 8932.JPG")

out = remove(img)
new_img_rgba = cv2.cvtColor(out, cv2.COLOR_BGRA2RGB)

gray = cv2.cvtColor(new_img_rgba,cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
points = cv2.findNonZero(thresh)

x, y, w, h = cv2.boundingRect(points)

top_left = (x, y)
top_right = (x + w, y)
bottom_left = (x, y + h)
bottom_right = (x + w, y + h)


plt.subplot(1, 2, 1)
plt.imshow(new_img_rgba)
plt.subplot(1, 2, 2)
new_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.rectangle(new_img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
plt.imshow(new_img_rgb)
plt.show()
