import numpy as np
import scipy.signal
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import cv2
import scipy.signal

d1_mask_gaussian = cv2.getGaussianKernel(35, sigma=5.5)
d2_mask_gaussian = np.outer(d1_mask_gaussian, np.transpose(d1_mask_gaussian))

d1_gaussian = cv2.getGaussianKernel(9, sigma=1.5)
d2_gaussian = np.outer(d1_gaussian, np.transpose(d1_gaussian))
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) # basic 4 neighbor

filepath_of_result = "./results/cat_x_dog.jpg"

##### generally use images that have resized on them. original images will have shape mismatches ######
def create_gaussian_stack(image, filter, stack):
    if len(stack) == 5:
        return stack
    else:
        blurred_im = scipy.signal.convolve2d(image, filter, mode='same')
        stack.append(blurred_im)
        return create_gaussian_stack(blurred_im, filter, stack)
    
def create_la_placian_stack(gaussian_stack):
    num = 0
    laplacian = []

    for num in range(0, len(gaussian_stack) - 1):
        image = gaussian_stack[num] - gaussian_stack[num + 1]
        if num % 2 == 0:
            displayable_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #cv2.imshow(f"laplacian level {num}", displayable_image)
            #cv2.waitKey(0)
        laplacian.append(gaussian_stack[num] - gaussian_stack[num + 1])

    laplacian.append(gaussian_stack[-1]) # append the last image in the gaussian, this one doesn't require any subtraction

    return laplacian


im_apple = cv2.imread("./images/apple.jpeg", cv2.IMREAD_GRAYSCALE) 
im_apple_float = im_apple.astype(np.float32)

GA = create_gaussian_stack(im_apple, d2_gaussian, [im_apple_float])
LA = create_la_placian_stack(GA)

im_orange = cv2.imread("./images/orange.jpeg", cv2.IMREAD_GRAYSCALE)
im_orange_float = im_orange.astype(np.float32)

GB = create_gaussian_stack(im_orange, d2_gaussian, [im_orange_float])
LB = create_la_placian_stack(GB)

# USE FOR THE LION AND CAT SPLINED IMAGE. 
# im_apple = cv2.imread("./images/cat_resized.jpg", cv2.IMREAD_GRAYSCALE) 
# im_apple_float = im_apple.astype(np.float32)

# GA = create_gaussian_stack(im_apple, d2_gaussian, [im_apple_float])
# LA = create_la_placian_stack(GA)

# im_orange = cv2.imread("./images/lion_resized.jpg", cv2.IMREAD_GRAYSCALE)
# im_orange_float = im_orange.astype(np.float32)

# GB = create_gaussian_stack(im_orange, d2_gaussian, [im_orange_float])
# LB = create_la_placian_stack(GB)

## USE FOR CREATION OF ADAM X ROBOT HAND
# im_apple = cv2.imread("./images/creation_of_adam_resized.jpg", cv2.IMREAD_GRAYSCALE) 
# im_apple_float = im_apple.astype(np.float32)

# GA = create_gaussian_stack(im_apple, d2_gaussian, [im_apple_float])
# LA = create_la_placian_stack(GA)

# im_orange = cv2.imread("./images/robot_hand_resized.jpeg", cv2.IMREAD_GRAYSCALE)
# im_orange_float = im_orange.astype(np.float32)

# GB = create_gaussian_stack(im_orange, d2_gaussian, [im_orange_float])
# LB = create_la_placian_stack(GB)



# depending on the shape of the image, you may need to add 1 mask_a or b's width/height
h, w = im_apple.shape

# This is for apple and orange
mask_a = np.ones((h, w // 2)) # Left half is ones
mask_b = np.zeros((h, w // 2)) # right half is zeroes


# FOR THE LION AND DOG!!!!

# mask_a = np.ones((h, 1 + w // 2)) # Left half is ones
# mask_b = np.zeros((h, w // 2)) # right half is zeroes


# final pixel = mask_pixel * imageA_pixel + ((1 - mask_pixel) * image_b pixel)
#concatenating for the straight line masks
mask = np.concat([mask_a, mask_b], axis=1)

# USE THIS MASK WHEN CREATING THE ROBOT HAND X CREATION OF ADAM
# mask = cv2.imread("./images/robot_hand_mask.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

# cv2.imshow("mask", mask)
# cv2.waitKey(0)

GR = create_gaussian_stack(mask, d2_mask_gaussian, [mask])

# cv2.imshow("mask level", GR[3])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(GR[0].shape, LB[0].shape, LA[0].shape)

LS = []
num = 0
Apple_stack = []
Orange_stack = []

for la_level, lb_level, gr_level in zip(LA, LB, GR):
    
    # apple image weighed
    apple_blended_level = la_level * gr_level
    apple_blended_level_displayable = cv2.normalize(apple_blended_level, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow(f"Apple with smoothing, level {num}", apple_blended_level_displayable)
    cv2.waitKey(0)
    Apple_stack.append(apple_blended_level)

    orange_blended_level = lb_level * (1 - gr_level)
    orange_blended_level_displayable = cv2.normalize(orange_blended_level, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow(f"orange with smoothing, level {num}", orange_blended_level_displayable)
    cv2.waitKey(0)
    Orange_stack.append(orange_blended_level)

    blended_level = la_level * gr_level + (1 - gr_level) * lb_level
    blended_level_displayable = cv2.normalize(blended_level, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow("output", blended_level_displayable)
    cv2.waitKey(0)
    
    LS.append(blended_level)
    num += 1


Apple_mask = np.sum(Apple_stack, axis=0)
Apple_mask_clipped = np.clip(Apple_mask, 0, 255).astype(np.uint8)
cv2.imshow("Apple Contribution", Apple_mask_clipped)
cv2.waitKey(0)

# Reconstruct the Orange contribution
orange_mask = np.sum(Orange_stack, axis=0)
orange_mask_clipped = np.clip(orange_mask, 0, 255).astype(np.uint8) # Convert to uint8
cv2.imshow("Orange Contribution", orange_mask_clipped)
cv2.waitKey(0)

splined_image = np.sum(LS, axis=0)
splined_image_clipped = np.clip(splined_image, 0, 255).astype(np.uint8)


cv2.imshow("Splined image", splined_image_clipped)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite(filepath_of_result, splined_image_normalized)











    


