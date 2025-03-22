import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images.jpg', cv2.IMREAD_COLOR)
n = 5  

# (a) 應用平均濾波器
average_filter_image = cv2.blur(image, (n, n))

# (b) 應用媒體濾波器
median_filter_image = cv2.medianBlur(image, n)

image_int16 = image.astype(np.int16)
average_filter_image_int16 = average_filter_image.astype(np.int16)
median_filter_image_int16 = median_filter_image.astype(np.int16)

average_image = cv2.subtract(image_int16, average_filter_image_int16)
median_image = cv2.subtract(image_int16, median_filter_image_int16)

k = 3.0  
sharpened_average_image = np.clip(image_int16 + k * average_image, 0, 255).astype(np.uint8)
sharpened_median_image = np.clip(image_int16 + k * median_image, 0, 255).astype(np.uint8)

plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(average_filter_image, cv2.COLOR_BGR2RGB))
plt.title(f"Average Filter ({n}x{n})")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(sharpened_average_image, cv2.COLOR_BGR2RGB))
plt.title(f"Unsharp Masking (Average)")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(median_filter_image, cv2.COLOR_BGR2RGB))
plt.title(f"Median Filter ({n}x{n})")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(sharpened_median_image, cv2.COLOR_BGR2RGB))
plt.title("Unsharp Masking (Median)")
plt.axis('off')

plt.tight_layout()
plt.show()
