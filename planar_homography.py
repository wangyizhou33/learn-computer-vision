import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

# load and display image
im = plt.imread("images/chess.PNG")
plt.imshow(im)
plt.axis("off")

# specify world coordinates of rectangular book cover
X = np.array([0, 200, 200, 0])
Y = np.array([200, 200, 0, 0])

# specify image coordinates of distorted book cover
# lower-left, lower-right, upper-right, upper-left
u = np.array([11.1, 347, 749.4, 400])
v = np.array([273.4, 624.3, 299.5, 100])

# display image with the image coordinates
plt.scatter(u, v, facecolors="none", edgecolors="y")
plt.show()

# estimate the homograph
A = np.zeros((8, 9))
for i in range(0, 4):
    A[2 * i, :] = [0, 0, 0, -X[i], -Y[i], -1, v[i] * X[i], v[i] * Y[i], v[i]]
    A[2 * i + 1, :] = [X[i], Y[i], 1, 0, 0, 0, -u[i] * X[i], -u[i] * Y[i], -u[i]]

# total least squares
L, V = la.eig(A.T @ A)
h = V[:, -1]  # minimal eigenvalue eigenvector
H = np.reshape(h, (3, 3))  # reshape into 3x3 homography
H = la.inv(H)

# rectify image based on homography
im_warp = cv2.warpPerspective(im, H, (200, 200))

# display rectified image
plt.imshow(im_warp)
plt.gca().invert_yaxis()
plt.axis("off")
plt.show()
