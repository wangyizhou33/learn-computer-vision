import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import sepfir2d

img = plt.imread("images/einstein.jpg")

p = [0.030320, 0.249724, 0.439911, 0.249724, 0.030320]
d = [-0.104550, -0.292315, 0.0, 0.292315, 0.104550]

img_x = sepfir2d(img, d, p)  # spatial (x) derivative
img_y = sepfir2d(img, p, d)  # spatial (y) derivative
img_g = np.sqrt(img_x**2 + img_y**2)  # gradient

plt.figure(figsize=(15, 4))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap="gray")
plt.title("original")
plt.subplot(1, 4, 2)
plt.imshow(img_x, cmap="gray")
plt.title("horizontal derivative")
plt.subplot(1, 4, 3)
plt.imshow(img_y, cmap="gray")
plt.title("vertical derivative")
plt.subplot(1, 4, 4)
plt.imshow(img_g, cmap="gray")
plt.title("gradient")
plt.show()
