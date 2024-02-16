import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sepfir2d

im = plt.imread("images/mandrill.png")
h = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]
N = 3

P = []
P.append(im)
for k in range(1, N):
    im2 = np.zeros(im.shape)
    for z in range(3):
        im2[:, :, z] = sepfir2d(im[:, :, z], h, h)  # blur each color channel
    im2 = im2[0:-1:2, 0:-1:2, :]  # down-sample
    im = im2
    P.append(im2)

# display pyramid
fig, ax = plt.subplots(
    nrows=1, ncols=N, figsize=(15, 7), dpi=72, sharex=True, sharey=True
)
for k in range(N - 1, -1, -1):
    ax[k].imshow(P[k])
plt.show()
