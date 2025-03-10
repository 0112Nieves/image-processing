import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

I = Image.open('cat_origin.jpg').convert('L')
I = np.array(I)

# 2-level dither
D2 = np.array([
    [0, 128, 32, 160],
    [192, 64, 224, 96],
    [48, 176, 16, 144],
    [240, 112, 208, 80]
])

height, width = I.shape
D = np.tile(D2, (height // 4 + 1, width // 4 + 1))[:height, :width]
I_2prime = np.where(I > D, 255, 0).astype(np.uint8)

# 4-level dither
D1 = np.array([
    [0, 56],
    [84, 28]
])
height, width = I.shape
D = np.tile(D1, (height // 2 + 1, width // 2 + 1))[:height, :width]
step = 85
Q = (I // step).astype(int)
I_4prime = Q + (I - step * Q > D).astype(int)
I_4prime = (I_4prime * step).astype(np.uint8)

Image.fromarray(I_2prime).save('cat_dithered_2_level.jpg')
Image.fromarray(I_4prime).save('cat_dithered_4_level.jpg')

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(I_2prime, cmap='gray')
plt.title('2-level Dithered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(I_4prime, cmap='gray')
plt.title('4-level Dithered Image')
plt.axis('off')

plt.show()