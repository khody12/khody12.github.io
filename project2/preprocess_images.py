# making images the same shape and some naive alignment. 
import numpy as np
import scipy.signal
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import cv2
import scipy.signal

# image1 = cv2.imread("./images/robot_hand.jpeg", cv2.IMREAD_GRAYSCALE)

# h, w = image1.shape
# image1 = cv2.resize(image1, (2 * w, 2 * h))

# cv2.imwrite("./images/robot_hand_resized.jpeg", image1)
# image1 = cv2.imread("./images/robot_hand_resized.jpeg", cv2.IMREAD_GRAYSCALE)

# cv2.imshow("image1", image1)
# print(image1.shape)
# image2 = cv2.imread("./images/creation_of_adam.jpg", cv2.IMREAD_GRAYSCALE)

# image2 = image2[90:, 63:671]
# print(image2.shape)
# cv2.imshow("image2", image2)

# cv2.imwrite("./images/creation_of_adam_resized.jpg", image2)

# mask = cv2.imread("./images/robot_hand_m.png", cv2.IMREAD_GRAYSCALE) / 255

# cv2.imshow("mask", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#275, 183
#300, 168

image1 = cv2.imread("./images/cat.jpeg", cv2.IMREAD_GRAYSCALE) # 275, 183
image2 = cv2.imread("./images/lion2.jpg", cv2.IMREAD_GRAYSCALE) # 300, 168
print(image1.shape, image2.shape)
image2 = image2[110: 660, 120:550]
image1 = image1[50:220, :]

cv2.imwrite("edited_cat.jpg", image1)

cv2.imshow("", image1)
cv2.waitKey(0)

cv2.imshow("", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

h1, w1 = image1.shape

resized_image2 = cv2.resize(image2, (w1, h1))
cv2.imshow("", resized_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("./images/resized_dog2.jpg", resized_image2)


