import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)

# 直方圖均衡化
equalized_image = cv2.equalizeHist(image)
cv2.imwrite('equalized_cat.jpg', equalized_image)

# 計算直方圖
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# 顯示原圖與均衡化後的影像
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.title("Original Grayscale")
plt.imshow(image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Equalized Grayscale")
plt.imshow(equalized_image, cmap='gray')

# 顯示直方圖
plt.subplot(2, 2, 3)
plt.title("Histogram of Original Image")
plt.plot(hist_original, color='black')

plt.subplot(2, 2, 4)
plt.title("Histogram of Equalized Image")
plt.plot(hist_equalized, color='black')

plt.tight_layout()
plt.show()
