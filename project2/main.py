import numpy as np
import scipy.signal
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import cv2
import scipy.signal

def read_and_resize(image_path):
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    

    print("old size: ", im.shape)

    h, w = im.shape
    
    im = cv2.resize(im, (int(w / 4), int(h / 4)), interpolation=cv2.INTER_AREA) # open_cv wants it in width, height
    # instead of numpy which wants it in height, width ... great...
    print("new size: ", im.shape)

    # cv2.imshow("resized image", im)
    # cv2.waitKey(0) # wait till u hit any key
    # cv2.destroyAllWindows() # close all the windows open cv made.
    return im

def four_loop_convolution(image, kernel):
    kernel = np.flip(kernel, axis = 0)
    kernel = np.flip(kernel, axis = 1)

    k_h, k_w = kernel.shape
    pad_h = (k_h - 1) // 2
    pad_w = (k_w - 1) // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w))) # np.pad adds 1-pixel border of zeros

    h, w = image.shape # get the shape of image we need to iterate over
    result = np.zeros((h, w))

    for height in range(0, h):
        for width in range(0, w):
            sum = 0
            for i in range(k_h):
                for j in range(k_h):
                    image_pixel = padded_image[height + i, width + j]
                    kernel_value = kernel[i, j]
                    sum += image_pixel * kernel_value
                    
                    
            result[height, width] = sum
    return result
            

    
def convolution(image, kernel):
    # first off, lets add padding to the image
    kernel = np.flip(kernel, axis = 0)
    kernel = np.flip(kernel, axis = 1)

    k_h, k_w = kernel.shape
    pad_h = (k_h - 1) // 2
    pad_w = (k_w - 1) // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w))) # np.pad adds 1-pixel border of zeros

    h, w = image.shape # get the shape of image we need to iterate over
    result = np.zeros((h , w))

    for height in range(0, h):
        for width in range(0, w):
            sum = np.sum(padded_image[height: height + k_h, width: width + k_w] * kernel)
            value = sum  # opencv needs it to be normalized to display

            result[height, width] = value
            
    return result

def compare_convolutions():
    box_filter = np.ones((9, 9))

    im = read_and_resize("./images/khody_image.JPG")

    convolved_khody = convolution(im, box_filter)
    convolved_khody = four_loop_convolution(im, box_filter)

    # normalize + box filter averaging
    convolved_khody /= (255 * 81)

    cv2.imshow("homemade_convolved", convolved_khody) # cv2 expects either integers from 0 -> 255, or floats from 0 -> 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #saveable_convolved_khody = cv2.normalize(convolved_khody, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imwrite("homemade_convolved_khody.png", saveable_convolved_khody)

    scipy_convolved_khody = scipy.signal.convolve2d(im, box_filter)

    cv2.imshow("scipy_convolved", scipy_convolved_khody / 255 / 81)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #saveable_scipy_convolved_khody = cv2.normalize(convolved_khody, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imwrite("scipy_convolved_khody.png", saveable_scipy_convolved_khody) saving image

compare_convolutions()

# Finite difference operators

Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])

def camera_man():
    im = cv2.imread("./images/cameraman.png", cv2.IMREAD_GRAYSCALE)

    X_partial_im = convolution(im, Dx) / 255 # normalizing with 255

    cv2.imshow("partial derivative wrt/x of cameraman", X_partial_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #saveable_dx = cv2.normalize(X_partial_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #cv2.imwrite("dx_cameraman.png", saveable_dx) saving to disk

    Y_partial_im = convolution(im, Dy) / 255

    cv2.imshow("partial derivative wrt/y of cameraman", Y_partial_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #saveable_dy = cv2.normalize(Y_partial_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #cv2.imwrite("dy_cameraman.png", saveable_dy) saving to disk

    grad_mag_im = np.sqrt(np.square(X_partial_im) + np.square(Y_partial_im))
    cv2.imshow("gradient magnitude image", grad_mag_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows


    binarize_im = np.where(grad_mag_im > 0.36, 255, 0).astype(np.uint8)
    cv2.imshow("binarized cameraman", binarize_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    saveable_binarize = cv2.normalize(binarize_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite("./results/binarized_cameraman.png", saveable_binarize)

camera_man()

def gaussian_cameraman():
    im = cv2.imread("./images/cameraman.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("original", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    d1_gaussian = cv2.getGaussianKernel(ksize=9, sigma=5)
    d2_gaussian = np.outer(d1_gaussian, np.transpose(d1_gaussian))

    blurred_im = convolution(im, d2_gaussian) # normalized
    normalized_blurred_im = cv2.normalize(blurred_im, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow("blurred image", normalized_blurred_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    X_partial_im = convolution(blurred_im, Dx) 
    normalized_X_partial_im = cv2.normalize(X_partial_im, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow("partial derivative wrt/x of cameraman", normalized_X_partial_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Y_partial_im = convolution(blurred_im, Dy)
    normalized_Y_partial_im = cv2.normalize(Y_partial_im, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow("partial derivative wrt/y of cameraman", normalized_Y_partial_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    grad_mag_im = np.sqrt(np.square(X_partial_im) + np.square(Y_partial_im))
    normalized_grad_mag_im = cv2.normalize(grad_mag_im, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow("gradient magnitude image", normalized_grad_mag_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    binarize_im = np.where(grad_mag_im > 25, 255, 0).astype(np.uint8)
    normalized_binarize_im = cv2.normalize(binarize_im, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # normalized_binarize_im = binarize_im / 255
    cv2.imshow("binarized cameraman", normalized_binarize_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./results/gaussian_binarized_cameraman.jpg", normalized_binarize_im)
gaussian_cameraman()

def DoG_filter():
    im = cv2.imread("./images/cameraman.png", cv2.IMREAD_GRAYSCALE)

    d1_gaussian = cv2.getGaussianKernel(ksize=9, sigma=5)
    d2_gaussian = np.outer(d1_gaussian, np.transpose(d1_gaussian))

    Gx = convolution(d2_gaussian, Dx)
    display_Gx = cv2.normalize(Gx, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    Gy = convolution(d2_gaussian, Dy)
    display_Gy = cv2.normalize(Gy, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    grad_x = convolution(im, Gx)
    display_grad_x = cv2.normalize(grad_x, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imwrite("./results/DoG_X.jpg", display_grad_x)

    grad_y = convolution(im, Gy)
    display_grad_y = cv2.normalize(grad_y, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite("./results/DoG_Y.jpg", display_grad_y)
    

    cv2.imshow("Gx filter", display_Gx)
    cv2.imshow("Gy Filter", display_Gy)
    

    cv2.imshow("Gradient X", display_grad_x)
    cv2.imshow("Gradient Y", display_grad_y)

    grad_mag_im = np.sqrt(np.square(grad_x) + np.square(grad_y))
    display_grad_mag = cv2.normalize(grad_mag_im, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow("grad mag im", display_grad_mag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




DoG_filter()
















    













            







    



