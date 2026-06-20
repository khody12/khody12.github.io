import numpy as np
import scipy.signal
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import cv2
import scipy.signal


d1_gaussian = cv2.getGaussianKernel(ksize=9, sigma=2.5)
d2_gaussian = np.outer(d1_gaussian, np.transpose(d1_gaussian))

# def sharpen():
#     im = cv2.imread("./images/taj.jpg")
#     cv2.imshow("original image", im)

#     # color image, so we need to convolve each channel

#     im_float = im.astype(np.float32) # typically want to do more complex math in float so that we aren't losing data

#     blurred_channels = []
#     for i in range(im_float.shape[2]):
#         channel = im_float[:, :, i]

#         blurred_channels.append(scipy.signal.convolve2d(channel, d2_gaussian, mode='same').astype(np.float32))

    
#     blurred_im = np.stack(blurred_channels, axis=2) # stack on axis = 2, which is a new axis. so lay them on top of each other
#     print(blurred_im.dtype)
#     # a problem we have here is that blurred images could have higher values than

#     # open cv uses brg. it loads in bgr. 
#     blurred_im = np.clip(blurred_im, 0, 255).astype(np.uint8)
#     cv2.imshow("smoothed image", blurred_im)

#     details = im_float - blurred_im
#     details = np.clip(details, 0, 255) # we should typically use clip if we want to make negative values 0 and we have other data
#     # that is important and we want to preserve. Normalize would stretch it, but clip just takes care of the extremes
#     # we want clip here because its possible blurred image is brighter in a certain pixel value, and subtracting could make the value negative
#     # which would cause problems
#     details_image = details.astype(np.uint8)
    
#     cv2.imwrite("./results/test.jpg", details_image)
#     cv2.imshow("details", details_image)
#     cv2.waitKey(0)
    

#     sharpened_image = im_float + details
#     sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
#     cv2.imshow("sharpened", sharpened_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# single convolution strategy
def sharpen():
    im = cv2.imread("./images/blurry_me.jpeg")
    cv2.imshow("original image", im)

    im_float = im.astype(np.float32)

    identity = np.zeros((9, 9))
    identity[4, 4] = 1 # create the identity kernel

    filter = identity + 5 * (identity - d2_gaussian)

    blurred_channels = []
    for i in range(im_float.shape[2]):
        channel = im_float[:, :, i]

        blurred_channels.append(scipy.signal.convolve2d(channel, filter, mode='same').astype(np.float32))
        
    sharpened_image = np.stack(blurred_channels, axis=2)
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

    cv2.imshow("sharpened image", sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("sharpened_taj.jpg", sharpened_image)

    # [I + alpha * (I - g)] * image
sharpen()
# sharpen()