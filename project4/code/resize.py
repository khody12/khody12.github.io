import numpy
import cv2

for i in range(0, 95):
    im_in = cv2.imread(f"./final_data/IMG_{8082 + i}.JPG")
    h, w, c = im_in.shape

    resize_im = cv2.resize(im_in, dsize=(w // 20, h // 20))
    cv2.imwrite(f"./final_data/im{i}.JPG", resize_im)
