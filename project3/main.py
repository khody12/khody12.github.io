import numpy as np

import numpy.linalg
import cv2 

from stacks import create_gaussian_stack, create_la_placian_stack, create_gaussian_stack_greyscale, d2_gaussian

#1st image correspondences, i.e. pano1
#correspondences = {"im1_name":"im1","im2_name":"im2","im1Points":[[633,232],[338,355],[693,178],[339,268],[676,282],[657,344],[420,279],[536,407],[333,96],[531,77],[611,105],[377,309],[439,299],[482,293],[515,288],[395,332],[469,314],[629,492],[634,257],[593,218],[401,254]],"im2Points":[[348,231],[43,374],[391,184],[45,268],[380,274],[367,325],[149,280],[269,395],[28,60],[258,83],[329,117],[97,315],[171,300],[217,290],[249,283],[119,338],[204,313],[349,458],[350,253],[317,218],[127,250]]}

#2nd image correspondences i.e. pano2
correspondences = {"im1_name":"im6","im2_name":"im7","im1Points":[[738,309],[639,259],[500,338],[467,346],[417,416],[378,271],[966,236],[884,520],[956,491],[404,631],[770,668],[821,676],[957,630],[943,684],[602,616],[691,629],[780,431],[839,378],[833,332]],"im2Points":[[451,319],[356,268],[205,346],[164,357],[103,439],[47,267],[631,259],[577,503],[629,474],[93,687],[484,654],[529,654],[632,594],[624,644],[322,632],[412,630],[490,431],[537,382],[531,340]]}
# rectificaiton image correspondences

# rupert correspondences
# correspondences = {"im1_name": "rupert", "im2_name": "top down rupert", "im1Points": [[281,463],[801,422],[369,1088],[801,964],[575,440],[796,690],[323,754],[612,1018]],
#   "im2Points": [[0,0],[600,0],[0,750],[600,750],[300,0],[600,375],[0,375],[300,750]]}

# pano3 correspondences
# correspondences = {"im1_name":"im8","im2_name":"im9","im1Points":[[299,216],[532,149],[411,169],[615,127],[564,480],[704,494],[385,462],[445,442],[861,449],[875,579],[758,282],[756,329],[759,110],[759,173],[515,262],[342,417],[325,453],[912,475],[488,426],[385,298]],"im2Points":[[17,205],[298,152],[162,160],[384,135],[338,492],[475,499],[129,485],[202,460],[609,447],[618,566],[520,297],[519,340],[518,137],[519,195],[285,265],[69,440],[52,480],[650,469],[251,440],[129,300]]}

def computeH(im1_pts, im2_pts):
    p = np.array(im1_pts)
    ones_col = np.ones((p.shape[0], 1))
    p = np.hstack([p, ones_col])

    p_prime = np.array(im2_pts)
    #p_prime = np.hstack([p_prime, ones_col])

    A_list = []

    # building out the rows
    for i in range(0, len(p)):
        row1 = [p[i, 0], p[i, 1], 1, 0, 0, 0, -p_prime[i, 0] * p[i, 0], -p_prime[i, 0] * p[i, 1]]
        row2 = [0, 0, 0, p[i, 0], p[i, 1], 1, -p_prime[i, 1] * p[i, 0], -p_prime[i, 1] * p[i, 1]]

        A_list.append(row1)
        A_list.append(row2)

    A_mat = np.array(A_list)

    print(A_mat)

    p_prime_flattened = np.transpose(p_prime.reshape(1, -1))

    h_mat = np.linalg.lstsq(A_mat, p_prime_flattened)[0] # access the actual solution. 
    h_mat = np.append(h_mat, 1).reshape(3, 3)

    print(h_mat)
    
    return h_mat

def warpImageNearestNeighbor(im, H):

    #pixel = image[y, x]   point = (x, y), pixel = image[row, col]

    height, width, _ = im.shape

    print("H shape", H.shape)

    H_inverse = np.linalg.inv(H)
    print(H_inverse)

    output_list = []

    top_left = [[0], [0], [1]]
    top_right = [[width - 1], [0], [1]]

    bottom_left = [[0], [height - 1], [1]]
    bottom_right = [[width - 1], [height - 1], [1]]

    # forward translation to get the bounds of our new image, "@" = matmul
    translated_tl = H @ top_left
    translated_tr = H @ top_right
    translated_bl = H @ bottom_left
    translated_br = H @ bottom_right

    #normalizing data with the w value 
    normalized_tl = translated_tl / translated_tl[2]
    normalized_tr = translated_tr / translated_tr[2]
    normalized_bl = translated_bl / translated_bl[2]
    normalized_br = translated_br / translated_br[2]

    min_height = np.floor(min(normalized_tl[1], normalized_tr[1], normalized_bl[1], normalized_br[1])) # get the y values that live at [1]
    max_height = np.ceil(max(normalized_tl[1], normalized_tr[1], normalized_bl[1], normalized_br[1]))

    min_width = np.floor(min(normalized_tl[0], normalized_tr[0], normalized_bl[0], normalized_br[0]))
    max_width = np.ceil(max(normalized_tl[0], normalized_tr[0], normalized_bl[0], normalized_br[0]))

    corners = [min_height, min_width, max_height, max_width]
    
    print("widths", max_width, min_width)
    canvas_width = int(max_width - min_width)
    canvas_height = int(max_height - min_height)

    row_offset = int(min_height)
    col_offset = int(min_width)
    print("height, width", canvas_height, canvas_width)
    for row in range(0, canvas_height): # new height for the new matrix, if there is some sort of shrinking or expanding, we need to iterate over that
        for col in range(0, canvas_width):
            # 0.0 is the center of a pixel

            x_new = col
            y_new = row

            x_world = x_new + col_offset # with respect to the origin which is the original image. 
            y_world = y_new + row_offset



            coordinate = np.array([[x_world + 0.5], [y_world + 0.5], [1]])

            result = np.matmul(H_inverse, coordinate)

            rounded_result = np.round(result / result[2])

            col_coord = int(rounded_result[0].item()) # coordinate in the original image
            row_coord = int(rounded_result[1].item())

            if 0 <= row_coord < height and 0 <= col_coord < width:
                output_list.append(im[row_coord, col_coord])
                continue

            output_list.append((0, 0, 0))
            
            

    # print(output_list)
    return np.array(output_list).reshape((canvas_height, canvas_width, 3)), corners 


row_offset = 0
col_offset = 0
def warpImageBilinear(im, H):
    height, width, _ = im.shape

    H_inverse = np.linalg.inv(H)

    output_list = []

    top_left = [[0], [0], [1]]
    top_right = [[width - 1], [0], [1]]

    bottom_left = [[0], [height - 1], [1]]
    bottom_right = [[width - 1], [height - 1], [1]]

    # forward translation to get the bounds of our new image, "@" = matmul
    translated_tl = H @ top_left
    translated_tr = H @ top_right
    translated_bl = H @ bottom_left
    translated_br = H @ bottom_right

    print("hello", translated_tl, translated_tr, translated_bl, translated_br)

    #normalizing data with the w value 
    normalized_tl = translated_tl / translated_tl[2]
    normalized_tr = translated_tr / translated_tr[2]
    normalized_bl = translated_bl / translated_bl[2]
    normalized_br = translated_br / translated_br[2]

    min_height = int(np.floor(min(normalized_tl[1], normalized_tr[1], normalized_bl[1], normalized_br[1]))) # get the y values that live at [1]
    max_height = int(np.ceil(max(normalized_tl[1], normalized_tr[1], normalized_bl[1], normalized_br[1])))

    min_width = int(np.floor(min(normalized_tl[0], normalized_tr[0], normalized_bl[0], normalized_br[0])))
    max_width = int(np.ceil(max(normalized_tl[0], normalized_tr[0], normalized_bl[0], normalized_br[0])))

    corners = [min_height, min_width, max_height, max_width]
    
    canvas_width = int(max_width - min_width)
    canvas_height = int(max_height - min_height)

    row_offset = int(min_height)
    col_offset = int(min_width)

    print("offsets", row_offset, col_offset)

    im = np.pad(im, ((1,1), (1,1), (0,0)), constant_values=0)

    for row in range(0, canvas_height): # new height for the new matrix, if there is some sort of shrinking or expanding, we need to iterate over that
        for col in range(0, canvas_width):
            # 0.0 is the center of a pixel
            x_new = col
            y_new = row

            x_world = x_new + col_offset # with respect to the origin which is the original image. 
            y_world = y_new + row_offset

            coordinate = np.array([[x_world + 0.5], [y_world + 0.5], [1]])

            result = np.matmul(H_inverse, coordinate) # this is going to be in terms of (x, y)

            result = result / result[2] # normalizing

            orig_px = np.array([result[0], result[1]]) # orig px is done all geo calcs, so keep it in x, y format

            if 0 <= result[1] < height and 0 <= result[0] < width: # comparing y to height, x to width

                top_left = np.floor(orig_px)
                top_right = np.array([top_left[0] + 1, top_left[1]])
                bottom_left = np.array([top_left[0], top_left[1] + 1])
                bottom_right = np.array([top_left[0] + 1, top_left[1] + 1])

                dx = orig_px[0] - top_left[0] 
                dy = orig_px[1] - top_left[1]  

                c_tl = im[int(top_left[1]), int(top_left[0])]
                c_tr = im[int(top_right[1]), int(top_right[0])]
                c_bl = im[int(bottom_left[1]), int(bottom_left[0])]
                c_br = im[int(bottom_right[1]), int(bottom_right[0])]

                top_val = c_tl * (1 - dx) + c_tr * dx
                bottom_val = c_bl * (1 - dx) + c_br * dx


                total_value = top_val * (1 - dy) + bottom_val * dy

                output_list.append(tuple(total_value))
            else:
                # print("hellur")
                output_list.append((0, 0, 0))

    return np.array(output_list).reshape((canvas_height, canvas_width, 3)), corners

def stitch(im1, im2, corners_warped_im): # typically warped image is im1, image be warp into is im2

    # corners = [min_height, min_width, max_height, max_width]

    min_y = corners_warped_im[0] 
    min_x = corners_warped_im[1] 
    max_y = corners_warped_im[2]
    max_x = corners_warped_im[3]


    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    final_max_x = max(max_x, w2)
    final_min_x = min(0, min_x)
    final_max_y = max(max_y, h2)
    final_min_y = min(0, min_y)

    im2_min_x = 0
    im2_min_y = 0
    im2_max_x = w2
    im2_max_y = h2

    overlap_min_x = max(min_x, im2_min_x)
    overlap_max_x = min(max_x, im2_max_x)
    overlap_min_y = max(min_y, im2_min_y)
    overlap_max_y = min(max_y, im2_max_y)

    gradient = np.linspace(1, 0, num=overlap_max_x - overlap_min_x)
    overlap_mask = np.tile(gradient, (int(overlap_max_y - overlap_min_y), 1))

    pano_width = final_max_x - final_min_x
    pano_height = final_max_y - final_min_y

    canvas_mask = np.zeros((pano_height, pano_width)).astype(np.float32)
    canvas_warpedim = np.zeros((pano_height, pano_width, 3))
    canvas_im2 = np.zeros((pano_height, pano_width, 3))

    # we're going to place the original image at an x offset and a y offset

    starting_x = abs(final_min_x)
    starting_y = abs(final_min_y)

    print(starting_x, starting_y)

    im_x = 0
    im_y = 0
    for i in range(starting_x, starting_x + w2):
        im_y = 0
        for j in range(starting_y, starting_y + h2):
            canvas_warpedim[j, i] = im2[im_y, im_x]
            im_y += 1
        im_x += 1

    gaussian_warped = create_gaussian_stack(canvas_warpedim, d2_gaussian, [canvas_warpedim.astype(np.float32)], 2)
    laplacian_warped = create_la_placian_stack(gaussian_warped)


    cv2.imshow("f", canvas_warpedim.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for i in range(0, w1):
        for j in range(0, h1):
            if tuple(im1[j, i]) == (0, 0, 0):
                continue
            else:
                canvas_im2[j, i] = im1[j, i]
                canvas_mask[j, i] = 1 # setimage 1 area to white


    # calculate the starting coordinates for the slice
    start_y = int(overlap_min_y - final_min_y)
    start_x = int(overlap_min_x - final_min_x)

    # calculate the ending coordinates
    end_y = start_y + overlap_max_y - overlap_min_y
    end_x = start_x + overlap_max_x - overlap_min_x

    gradient = np.linspace(1, 0, num=(overlap_max_x - overlap_min_x))
    overlap_mask = np.tile(gradient, (overlap_max_y - overlap_min_y, 1))
    canvas_mask[start_y:end_y, start_x:end_x] = overlap_mask

    print(start_y, end_y, start_x, end_x)
    print(canvas_im2.shape, canvas_warpedim.shape, canvas_mask.shape)

    # correcting for the corners where we want all of the regular image or warped image but the mask is rectangular and cuts them off
    for i in range(start_y, end_y):
        for j in range(start_x, end_x):
            if tuple(canvas_im2[i, j]) == (0, 0, 0) and tuple(canvas_warpedim[i, j]) == (0, 0, 0):
                print("what was there before", canvas_mask[i, j])
                canvas_mask[i, j] = 0.0
            elif tuple(canvas_im2[i, j]) != (0, 0, 0) and tuple(canvas_warpedim[i, j]) == (0, 0, 0):
                canvas_mask[i, j] = 1
            elif tuple(canvas_im2[i, j]) == (0, 0, 0) and tuple(canvas_warpedim[i, j]) != (0, 0, 0):
                canvas_mask[i, j] = 0
            
        
    canvas_mask_normalized = cv2.normalize(canvas_mask, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    

    cv2.imshow("f", canvas_mask_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gaussian_canvas_mask = create_gaussian_stack_greyscale(canvas_mask, d2_gaussian, [canvas_mask], 2)
    

    gaussian_im2 = create_gaussian_stack(canvas_im2, d2_gaussian, [canvas_im2.astype(np.float32)], 2)
    laplacian_im2 = create_la_placian_stack(gaussian_im2)

    result_stack = []

    for i in range(len(laplacian_im2)):
        mask_level_2d = gaussian_canvas_mask[i]

        mask_level_3d = mask_level_2d[:, :, np.newaxis]
        
        result_stack.append((mask_level_3d) * laplacian_im2[i] + (1 - mask_level_3d) * laplacian_warped[i])
    
    result_image = np.sum(result_stack, axis=0)
    result_image_clipped = np.clip(result_image, 0, 255).astype(np.uint8)

    print(result_image_clipped.shape)

    cv2.imshow("result", result_image_clipped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("f", canvas_im2.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_image

if __name__ == "__main__":
    # MUST BE CHANGED DEPENDING ON THE IMAGES YOU ARE LOADING IN !!!!!!!!!!!!!!!!!
    im1 = cv2.imread("./images/im6.JPG")
    im2 = cv2.imread("./images/im7.JPG")

    homography = computeH(correspondences["im1Points"], correspondences["im2Points"])
    print("homography", homography)

    # uncomment if you want to test the interpolations, uncommented for speed purposes if you just want to go straight to testing the mosaics
    # image_prime = warpImageNearestNeighbor(im1, homography).astype(np.uint8)
    # cv2.imshow("image after nearest neighbor homography", image_prime) # needs uint8
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
            
    # image_prime, corners = warpImageBilinear(im1, homography)
    # print("corners", corners)
    # cv2.imshow("image after bilinear homography", image_prime.astype(np.uint8)) # needs uint8
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image_prime = cv2.imread("./images/warped_bilinear_im6.png")  # ALSO NEEDS TO BE CHANGED DEPENDING ON WHAT YOU ARE LOADING IN, IF YOU ARE RUNNING IT ALL AT ONCE, ITS OK. 
    # WARPED_BILINEAR_IM1.PNG FOR THE FIRST PANO ^^^^^^^^^
    # WARPED_BILINEAR_IM6.PNG FOR THE SECOND PANO
    # WARPED_BILINEAR_IM8.PNG FOR THE THIRD PANO

    # cv2.imwrite("./images/warped_bilinear_im8.png", image_prime)

    #corners = [-274, -818, 841, 410] # corners for pano1

    corners = [-175, -649, 1008, 671] # corners for pano2
    # corners = [-128, -499, 950, 725] # corners for pano3
    stitch(image_prime, im2, corners)
        