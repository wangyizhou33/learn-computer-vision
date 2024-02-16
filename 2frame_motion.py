import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from numpy import linalg
from scipy.signal import sepfir2d
from scipy import ndimage

# load two images
vid = []
vid.append(plt.imread("images/basketball1.png"))
vid.append(plt.imread("images/basketball2.png"))

# compute space-time derivative
p = [0.030320, 0.249724, 0.439911, 0.249724, 0.030320]
d = [-0.104550, -0.292315, 0.0, 0.292315, 0.104550]

fx = sepfir2d(0.5 * vid[0] + 0.5 * vid[1], d, p)
fy = sepfir2d(0.5 * vid[0] + 0.5 * vid[1], p, d)
ft = sepfir2d(vid[0] - vid[1], p, p)

h = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]
fx2 = sepfir2d(fx * fx, h, h)
fy2 = sepfir2d(fy * fy, h, h)
fxy = sepfir2d(fx * fy, h, h)
fxt = sepfir2d(fx * ft, h, h)
fyt = sepfir2d(fy * ft, h, h)

# compute motion (at every other pixel)
ydim, xdim = vid[0].shape
Vx = np.zeros((ydim // 2, xdim // 2))
Vy = np.zeros((ydim // 2, xdim // 2))

cx = 0

for x in range(0, xdim - 1, 2):
    cy = 0
    for y in range(0, ydim - 1, 2):
        # build the linear system
        M = [[fx2[y, x], fxy[y, x]], [fxy[y, x], fy2[y, x]]]
        b = [[fxt[y, x]], [fyt[y, x]]]

        # print(linalg.cond(M), np.sqrt(fx[y, x] * fx[y, x] + fy[y, x] * fy[y, x]) )

        if linalg.cond(M) < 200 and (
            np.sqrt(fx[y, x] * fx[y, x] + fy[y, x] * fy[y, x]) > 0.07
        ):
            v = -linalg.inv(M) @ b
            Vx[cy, cx] = v[0]
            Vy[cy, cx] = v[1]

        cy = cy + 1
    cx = cx + 1


# display the motion
plt.figure(figsize=(4 * xdim / 72, 4 * ydim / 72))
plt.imshow(vid[0], cmap="gray")

Ny, Nx = vid[0].shape
X, Y = np.meshgrid(np.arange(0, Nx - 1, 2), np.arange(0, Ny - 1, 2))
Vx = ndimage.median_filter(Vx, size=4)  # remove outliers
Vy = ndimage.median_filter(Vy, size=4)  # remove outliers
Vy = -1 * Vy  # flip y-axis

_ = plt.quiver(X, Y, Vx, Vy, scale=100, color="y", alpha=0.8, width=0.002, minlength=0)
plt.axis("off")
plt.show()
