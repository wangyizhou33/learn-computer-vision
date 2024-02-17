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
    f"Detected num of features = {num_features}, each feature vector has dimensions = {feature_dim}"
)

# Use BRIEF feature descriptor
# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# compute the descriptors with BRIEF
# which is binary string descriptors
kp_brief, des_brief = brief.compute(img, kp)
print(f"BRIEF has a reduced feature dimensions = {des_brief.shape[1]}")

plt.imshow(img)
plt.show()
