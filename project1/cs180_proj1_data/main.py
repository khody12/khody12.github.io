import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.filters import sobel
from PIL import Image
import sys

# using basic euclidean distance to compare
def align(channel_one, channel_two): # (base channel, channel we are aligning to the other)
    closest = 1000000000000
    displacement_h = 0
    displacement_w = 0
    for height in range(-15, 15): # search and see which combination of displacements leads to the best alignment
        for width in range(-15, 15):
            c_2 = np.roll(channel_two, height, axis=0)
            c_2 = np.roll(c_2, width, axis=1)
            res = np.sqrt(np.sum(np.square(channel_one - c_2)))

            if res < closest:
                closest = res
                displacement_h = height
                displacement_w = width

    
    aligned_h = np.roll(channel_two, displacement_h, axis=0)
    aligned = np.roll(aligned_h, displacement_w, axis=1)

    return aligned


            

# name of the input file
filenames = ["cathedral.jpg", "church.tif", "emir.tif", "harvesters.tif", "icon.tif", 
              "italil.tif", "lastochikino.tif", "lugano.tif", "melons.tif", "monastery.jpg", "self_portrait.tif", "siren.tif", "three_generations.tif", "tobolsk.jpg"]
filename_prefixes = ["cathedral", "church", "emir", "harvesters", "icon", 
              "italil", "lastochikino", "lugano", "melons", "monastery", "self_portrait", "siren", "three_generations", "tobolsk"]

# imname = 'emir.tif'

# ###   NAIVE IMPLEMENTATION   ###
im = skio.imread("emir.tif")

im = sk.img_as_float(im)

height = np.floor(im.shape[0] / 3.0).astype(np.int16)

b = im[:height] # define axis 0 and then just get all of axis 1. numpy infers you want all of it im[:height] = im[:height, :]
g = im[height: 2*height]
r = im[2*height: 3*height]

im_out = np.dstack([r, g, b])

plt.imshow(im_out, cmap="gray") # cmap stands for color map. translate numerical data into colors. typically we just use grey here. 
plt.title("Naive implementation")
plt.show()
# ###   NAIVE IMPLEMENTATION   ###


#adding with single-scale aligned implementation - slow version + crop.

crop = 0.1
h, w = b.shape

crop_h = int(h * crop)
crop_w = int(w * crop)

b = b[crop_h: h - crop_h, crop_w: w - crop_w]
g = g[crop_h: h - crop_h, crop_w: w - crop_w]
r = r[crop_h: h - crop_h, crop_w: w - crop_w]

aligned_red = align(b, r)
aligned_green = align(b, g)

im_out = np.dstack([aligned_red, aligned_green, b])

plt.imshow(im_out, cmap="gray") # cmap stands for color map. translate numerical data into colors. typically we just use grey here. 
plt.title("Single alignment + crop")
plt.show()


### IMAGE PYRAMID IMPLEMENTATION + EDGE MAP FOR EMIR###
###### !!!!!!!! USES SOBEL TO ALIGN EMIR! IF YOU WANT TO TEST SIMPLY L2 IMAGE PYRAMID IMPLEMENTATION, DELETE SOBEL FUNCTION CALLS WITHIN image_pyramid FUNCTION BELOW!!!! #####
# this will do the image pyramid on every image that was required. 

for num in range(0, len(filenames)):
    # read in the image
    im = skio.imread(filenames[num])

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int16)

    # separate color channels
    b = im[:height] 
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    crop = 0.1
    h, w = b.shape

    crop_h = int(h * crop)
    crop_w = int(w * crop)

    b = b[crop_h: h - crop_h, crop_w: w - crop_w]
    g = g[crop_h: h - crop_h, crop_w: w - crop_w]
    r = r[crop_h: h - crop_h, crop_w: w - crop_w]

    b = b * 255
    g = g * 255
    r = r * 255

    def image_pyramid(ip_ch1, ip_ch2):

        channel_one = ip_ch1.pop(0)
        channel_two = ip_ch2.pop(0)
        if len(ip_ch1) == 0:
            h, w = align(0, 0, sobel(channel_one), sobel(channel_two))
        else:
            h, w = image_pyramid(ip_ch1, ip_ch2)
            h_guess = 2 * h
            w_guess = 2 * w

            h, w = align(h_guess, w_guess, sobel(channel_one), sobel(channel_two))
        return h, w


    # taking in the height width located by the previous image and scanning starting form there
    def align(h, w, channel_one, channel_two):
        closest = 100000000000
        for height in range(h - 5, h + 5):
                for width in range(w - 5, w + 5):
                    c_2 = np.roll(channel_two, height, axis=0)
                    c_2 = np.roll(c_2, width, axis=1)
                    res = np.sqrt(np.sum(np.square(channel_one - c_2)))

                    if res < closest:
                        closest = res
                        displacement_h = height
                        displacement_w = width
        return displacement_h, displacement_w

    # creating lists for image pyramid recursion
    image_pyramid_blue_channel = []
    image_pyramid_blue_channel_2 = []
    image_pyramid_green_channel = []
    image_pyramid_red_channel = []

    image_pyramid_blue_channel.append(b)
    image_pyramid_blue_channel_2.append(b)
    image_pyramid_green_channel.append(g)
    image_pyramid_red_channel.append(r)

    h, _ = b.shape

    rescale_b = b
    rescale_g = g
    rescale_r = r
    while h > 40: # while the image is reasonably sized, scale it down
        rescale_b = sk.transform.rescale(rescale_b, 0.5, anti_aliasing=True)
        rescale_g = sk.transform.rescale(rescale_g, 0.5, anti_aliasing=True)
        rescale_r = sk.transform.rescale(rescale_r, 0.5, anti_aliasing=True)
        image_pyramid_blue_channel.append(rescale_b)
        image_pyramid_blue_channel_2.append(rescale_b)
        image_pyramid_green_channel.append(rescale_g)
        image_pyramid_red_channel.append(rescale_r)
        h, _ = rescale_b.shape

    # image pyramid with blue and green, then blue and red
    h, w = image_pyramid(image_pyramid_blue_channel, image_pyramid_green_channel)

    aligned_h = np.roll(g, h, axis=0)
    aligned_green = np.roll(aligned_h, w, axis=1)

    h, w = image_pyramid(image_pyramid_blue_channel_2, image_pyramid_red_channel)

    aligned_h = np.roll(r, h, axis=0)
    aligned_red = np.roll(aligned_h, w, axis=1)

    # align the aligned red, the aligned green, with blue
    im_out = np.dstack([aligned_red, aligned_green, b])
    im_out = im_out.astype(np.uint8)

    image = Image.fromarray(im_out, 'RGB')

    #image.save(filename_prefixes[num] + "_restored" + ".jpg")


    plt.imshow(im_out, cmap="gray") # cmap stands for color map. translate numerical data into colors. typically we just use grey here. 
    plt.title("pyramid implementation")
    plt.show()

