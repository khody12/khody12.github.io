import matplotlib.pyplot as plt
from align_image_code import align_images
import cv2
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

im1 = plt.imread('./images/DerekPicture.jpg')



# plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(cv2.cvtColor(im1.astype(np.float32), cv2.COLOR_RGB2GRAY))))))
# plt.title("Fourier Transform of low pass image")
# plt.show()

im2 = plt.imread('./images/nutmeg.jpg') #/255


# plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(cv2.cvtColor(im2.astype(np.float32), cv2.COLOR_RGB2GRAY))))))
# plt.title("Fourier Transform of the high pass image")
# plt.show()

im1 = im1 / 255
im2 = im2 / 255

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

def low_pass(image, sigma):
    d1_gaussian = cv2.getGaussianKernel(6 * sigma + 1, sigma)
    d2_gaussian = np.outer(d1_gaussian, np.transpose(d1_gaussian))


    im_float = image.astype(np.float32)

    low_pass_image = scipy.signal.convolve2d(im_float, d2_gaussian, mode='same')
    normalized_low_pass_image = cv2.normalize(low_pass_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)


    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(low_pass_image)))))
    plt.title("Fourier Transform with a low pass filter applied")
    plt.show()

    cv2.imshow("Low_pass", normalized_low_pass_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return low_pass_image # returns uint 8

def high_pass(image, sigma):

    gaussian_filtered = low_pass(image, sigma) # gaussian filter is uint8
    high_pass_image = image.astype(np.float32) - gaussian_filtered
    normalized_high_pass_image = cv2.normalize(high_pass_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(high_pass_image)))))
    plt.title("Fourier Transform with a high pass filter applied")
    plt.show()


    cv2.imshow("high pass", normalized_high_pass_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return high_pass_image


low_pass_image = low_pass(cv2.cvtColor(im1_aligned.astype(np.float32), cv2.COLOR_RGB2GRAY), 6) 
high_pass_image = high_pass(cv2.cvtColor(im2_aligned.astype(np.float32), cv2.COLOR_RGB2GRAY), 6) 

hybrid = low_pass_image + high_pass_image
hybrid_normalized = cv2.normalize(hybrid, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid)))))
plt.title("Fourier Transform of the hybrid image")
plt.show()


cv2.imshow("hybrid", hybrid_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("./results/test.png", hybrid_normalized)


# ## Compute and display Gaussian and Laplacian Pyramids
# ## You also need to supply this function
# N = 5 # suggested number of pyramid levels (your choice)
# pyramids(hybrid, N)