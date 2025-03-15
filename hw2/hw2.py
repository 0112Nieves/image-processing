import cv2
import numpy as np
import matplotlib.pyplot as plt

gray_image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread('landscape.jpg')

if gray_image is None or color_image is None:
    print("Error: Image not found!")
    exit()

### 灰階圖
equalized_gray = cv2.equalizeHist(gray_image)

hist_gray_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_gray_equalized = cv2.calcHist([equalized_gray], [0], None, [256], [0, 256])

### 彩圖
gray_from_color = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
equalized_gray_from_color = cv2.equalizeHist(gray_from_color)

# 避免除0
gray_from_color = np.maximum(gray_from_color, 1)

# 計算新的 RGB 值 (r’, g’, b’) = (r, g, b) * G’ / G
color_image_float = color_image.astype(np.float32)
equalized_gray_from_color_float = equalized_gray_from_color.astype(np.float32)
gray_from_color_float = gray_from_color.astype(np.float32)

new_color_image = (color_image_float * (equalized_gray_from_color_float[:, :, None] / gray_from_color_float[:, :, None]))
new_color_image = np.clip(new_color_image, 0, 255).astype(np.uint8)

hist_color_original_r = cv2.calcHist([color_image], [2], None, [256], [0, 256])
hist_color_original_g = cv2.calcHist([color_image], [1], None, [256], [0, 256])
hist_color_original_b = cv2.calcHist([color_image], [0], None, [256], [0, 256])

hist_color_equalized_r = cv2.calcHist([new_color_image], [2], None, [256], [0, 256])
hist_color_equalized_g = cv2.calcHist([new_color_image], [1], None, [256], [0, 256])
hist_color_equalized_b = cv2.calcHist([new_color_image], [0], None, [256], [0, 256])

plt.figure(figsize=(15, 5))

plt.subplot(2, 3, 1)
plt.title("Original Grayscale")
plt.imshow(gray_image, cmap='gray')

plt.subplot(2, 3, 2)
plt.title("Equalized Grayscale")
plt.imshow(equalized_gray, cmap='gray')

plt.subplot(2, 3, 3)
plt.title("Histogram (Grayscale)")
plt.plot(hist_gray_original, color='black', label='Original')
plt.plot(hist_gray_equalized, color='red', label='Equalized')
plt.legend()

plt.subplot(2, 3, 4)
plt.title("Original Color Image")
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 5)
plt.title("Equalized Color Image")
plt.imshow(cv2.cvtColor(new_color_image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 6)
plt.title("Histogram (Color)")
plt.plot(hist_color_original_r, color='r', linestyle='dotted', label='Original R')
plt.plot(hist_color_original_g, color='g', linestyle='dotted', label='Original G')
plt.plot(hist_color_original_b, color='b', linestyle='dotted', label='Original B')
plt.plot(hist_color_equalized_r, color='r', label='Equalized R')
plt.plot(hist_color_equalized_g, color='g', label='Equalized G')
plt.plot(hist_color_equalized_b, color='b', label='Equalized B')
plt.legend()

plt.tight_layout()
plt.show()

cv2.imwrite('equalized_cat.jpg', equalized_gray)
cv2.imwrite('equalized_landscape.jpg', new_color_image)