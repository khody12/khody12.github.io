import numpy
import cv2

# for i in range(0, 40):
#     im_in = cv2.imread(f"./nerf_data_images_2/IMG_{7666 + i}.JPG")
#     h, w, c = im_in.shape

#     resize_im = cv2.resize(im_in, dsize=(w // 20, h // 20))
#     cv2.imwrite(f"./nerf_data_images_2/IMG_{7666 + i}.JPG", resize_im)
for i in range(0,40):
    im_in = cv2.imread(f"./nerf_data_images_2/IMG_{7666 + i}.JPG")
    cv2.imwrite(f"./nerf_data_images_2/im{i}.JPG", im_in)