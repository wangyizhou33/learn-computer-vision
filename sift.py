import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("images/home.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

num_features, feature_dim = des.shape

print(
    f"Detected num of features = {num_features}, each feature vector has {feature_dim} dimensions"
)

plt.imshow(img)
plt.show()
