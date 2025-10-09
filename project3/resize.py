import cv2


im1 = cv2.imread("./images/im8.JPG")

im1_resized = cv2.resize(im1, None, fx=0.25, fy=0.25)

cv2.imwrite("im8.JPG", im1_resized)

im2 = cv2.imread("./images/im9.JPG")

im2_resized = cv2.resize(im2, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

cv2.imwrite("im9.JPG", im2_resized)