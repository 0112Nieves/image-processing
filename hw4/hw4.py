import cv2
import numpy as np
import matplotlib.pyplot as plt

size = 200
image = np.zeros((size, size), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (150, 150), 255, -1)

angle = 30
center = (size // 2, size // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)

# nearest neighbor interpolation
rotated_nearest = cv2.warpAffine(image, M, (size, size), flags=cv2.INTER_NEAREST)

# bilinear interpolation
rotated_bilinear = cv2.warpAffine(image, M, (size, size), flags=cv2.INTER_LINEAR)

cv2.imwrite("original.png", image)                # 儲存原始影像
cv2.imwrite("rotated_nearest.png", rotated_nearest)  # 儲存最近鄰插值結果
cv2.imwrite("rotated_bilinear.png", rotated_bilinear)  # 儲存雙線性插值結果

# 顯示結果
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(image, cmap='gray')
axs[0].set_title("Original Image")
axs[1].imshow(rotated_nearest, cmap='gray')
axs[1].set_title("Nearest Neighbor Rotation")
axs[2].imshow(rotated_bilinear, cmap='gray')
axs[2].set_title("Bilinear Rotation")

for ax in axs:
    ax.axis('off')

plt.show()
