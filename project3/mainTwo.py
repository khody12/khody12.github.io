import numpy as np

import numpy.linalg
import cv2 


from main import warpImageBilinear, stitch

import skimage as skimage

from stacks import create_gaussian_stack, create_la_placian_stack, create_gaussian_stack_greyscale
from scipy.spatial.distance import cdist

from harris import get_harris_corners, dist2

# from project3a
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


# single scale harris interest point detector

im = cv2.imread("./images/im1.JPG")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

harris_im, harris_corners = get_harris_corners(im_gray)

im_copy = im.copy()
for point in zip(harris_corners[0], harris_corners[1]):
    r, c = point

    center_xy = (c, r) # x, y for opencv

    cv2.circle(im_copy, center_xy, 3, [0,0,255,], 1)

cv2.imshow("harris image", im_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()


def adaptive_non_max_suppression(im_h, orig_im):
    # lets only consider points which are somewhat reasonably bright
    # some limit of points
    MAX_POINTS_BEFORE_ANMS = 20000

    # get all scores from the image
    all_strengths = im_h.flatten()

    # if we have more points than our limit
    if len(all_strengths) > MAX_POINTS_BEFORE_ANMS:
        # get Nth largest value.
        all_strengths.partition(-MAX_POINTS_BEFORE_ANMS)
        threshold = all_strengths[-MAX_POINTS_BEFORE_ANMS]
    else:
        threshold = im_h.min()

    if threshold <= 0.0:
        threshold = 0.01 * im_h.max()

    rows, cols = np.where(im_h > threshold) # np.where(condition) returns rows, cols
    # returns 2 1d arrays, one for y, another for x. 
    # np.where(condition, x, y) returns distances

    points = np.column_stack((rows, cols)) # we now have an N x 2 matrix, y, x. one column is x's the other is y's
    strengths = im_h[rows, cols] # gets the strength for each coordinate pair, each (row, col), at index 0 it takes one of the row, and another from the col

    N = len(strengths)

    robust_factor = 0.9

    # create all the possible distances between the points
    dist_matrix = dist2(points, points)
    # np will compare a column to a row, broadcasts it into nxn, and will compare every elem in column to every elem in row
    is_stronger_neighbor = strengths.reshape(N, 1) < (robust_factor * strengths.reshape(1, N))
    # print(is_stronger_neighbor)

    # distances[i, j] will contain the squared distance 
    distances = np.where(is_stronger_neighbor, dist_matrix, np.inf)

    radii = np.min(distances, axis=1)
    sorting_indices = np.argsort(radii) # argsort returns the indices that sort the array, basically picks out the elements from the original array based on the sort

    descending_indices = sorting_indices[::-1]

    best_indices = descending_indices[0:500]

    final_points = points[best_indices]
    final_strengths = strengths[best_indices]

    height, width = im_h.shape

    new_harris_im = np.zeros((height, width))
    for point, strength in zip(final_points, final_strengths):
        new_harris_im[point[0], point[1]] = strength

    im_with_corners = orig_im.copy()

    for point in final_points:
        r, c = point

        center_xy = (c, r)
        cv2.circle(im_with_corners, center_xy, 3, [0, 0, 255], 1)
    # cv2.imshow("image with corners ", im_with_corners)
    # cv2.waitKey(0)
    
    return new_harris_im, final_points
    

def feature_descriptor_extraction(im, feature_points):
    # cv2.imshow("input image", im)

    blurred_im = cv2.GaussianBlur(im, (9, 9), 1.5)
    
    padded_im = np.pad(blurred_im, 20, mode='constant', constant_values=0)
    # cv2.imshow("padded image", padded_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    num = 0
    feature_descriptors = []

    for point in feature_points:
        num += 1
        r, c = point[0] + 20, point[1] + 20
        window = padded_im[r - 20: r + 20, c - 20: c + 20]

        feature_descriptor = skimage.transform.resize(window, (8, 8), anti_aliasing=True)
        
        # true_normalizaiton
        mean = np.mean(feature_descriptor)
        std = np.std(feature_descriptor)

        # if num < 10:
        #     display_patch = cv2.resize(feature_descriptor, (160, 160), interpolation=cv2.INTER_NEAREST)
        #     display_patch = cv2.normalize(display_patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #     cv2.imshow("feature", display_patch)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        #     cv2.imshow("window", window)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        if std > 0:
            normalized_feature_descriptor = (feature_descriptor - mean) / std # maintaing mean and std, we use this for matching features
        feature_descriptors.append(normalized_feature_descriptor.flatten())

    return np.array(feature_descriptors)

    #return feature_descriptors

def feature_matching(im1, im2): # now we'll work to actually feature match.
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    harris_im1, _ = get_harris_corners(im1_gray)
    harris_im2, _ = get_harris_corners(im2_gray)

    final_harris_im1, final_points_im1 = adaptive_non_max_suppression(harris_im1, im1)
    final_harris_im2, final_points_im2 = adaptive_non_max_suppression(harris_im2, im2)

    feature_descriptors_im1 = feature_descriptor_extraction(im1_gray, final_points_im1)
    feature_descriptors_im2 = feature_descriptor_extraction(im2_gray, final_points_im2)

    print(feature_descriptors_im1.shape, feature_descriptors_im2.shape)

    # features_descriptors_im1 = np.array([feature_descriptors_im1])

    distance_matrix = cdist(feature_descriptors_im1, feature_descriptors_im2, metric="sqeuclidean")
    N = len(final_points_im1)
    # i = im1, j = im2

    sorted_indices = np.argsort(distance_matrix, axis=1) # indices with smallest distance across row

    best_match_indices = sorted_indices[:, 0] # grabbing the best matches for im i
    second_best_match_indices = sorted_indices[:, 1] # grabbing second best matches

    d1 = distance_matrix[np.arange(N), best_match_indices]
    d2 = distance_matrix[np.arange(N), second_best_match_indices]
    
    ratio = 0.8

    good_matches = (d1 / d2) < ratio # d1 is significantly closer check

    indices_im1 = np.where(good_matches)[0]
    indices_im2 = best_match_indices[good_matches]

    coordinates_im1 = final_points_im1[indices_im1]
    coordinates_im2 = final_points_im2[indices_im2]

    keypoints1 = []
    keypoints2 = []
    # get everything into keypoint format to create our lines in the matching image
    for k in range(len(coordinates_im1)):
        kp1 = cv2.KeyPoint(x=float(coordinates_im1[k][1]), y=float(coordinates_im1[k][0]), size=3)
        kp2 = cv2.KeyPoint(x=float(coordinates_im2[k][1]), y=float(coordinates_im2[k][0]), size=3)
        keypoints1.append(kp1)
        keypoints2.append(kp2)

    # create dmatch objects
    dmatches = []
    for k in range(len(keypoints1)):
        match = cv2.DMatch(k, k, 0) 
        dmatches.append(match)

    output_image = None
    output_image = cv2.drawMatches(
        im1,            
        keypoints1,     
        im2,            
        keypoints2,     
        dmatches,       
        output_image,   
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS 
    )

    # 4. Display the result
    cv2.imshow("Matches", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # padded_im1_greyscale = np.pad(im1_gray, 20, mode='constant', constant_values=0)
    # padded_im2_greyscale = np.pad(im2_gray, 20, mode='constant', constant_values=0)
    # for i in range(0, 5):

    #     r, c = coordinates_im1[i][0] + 20, coordinates_im1[i][1] + 20
    #     window_1 = padded_im1_greyscale[r - 20: r + 20, c - 20: c + 20]

    #     r2, c2 = coordinates_im2[i][0] + 20, coordinates_im2[i][1] + 20
    #     window_2 = padded_im2_greyscale[r2 - 20: r2 + 20, c2 - 20: c2 + 20]

    #     descriptor_1 = feature_descriptors_im1[indices_im1[i]].reshape((8, 8))
    #     descriptor_2 = feature_descriptors_im2[indices_im2[i]].reshape((8, 8))

    #     display_patch_1 = cv2.resize(descriptor_1, (160, 160), interpolation=cv2.INTER_NEAREST)
    #     display_patch_1 = cv2.normalize(display_patch_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     cv2.imshow("feature 1 descriptor", display_patch_1)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    #     display_patch_2 = cv2.resize(descriptor_2, (160, 160), interpolation=cv2.INTER_NEAREST)
    #     display_patch_2 = cv2.normalize(display_patch_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     cv2.imshow("feature 2 descriptor", display_patch_2)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    #     cv2.imshow("window from image 1", window_1)
    #     cv2.imshow("feature from image 2", window_2)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return coordinates_im1, coordinates_im2


def ransac(points_im1, points_im2, min_sample_size, num_iterations=5000, inlier_threshold=5):
    K = len(points_im1)

    best_h = None
    max_inliers = -1
    best_inlier_mask = None # tells us which of the original matches are solid and which are bad
    num = 0
    while num < num_iterations:
        

        curr_inlier_mask = [False] * K
        current_inliers = 0
        sample_indices = np.random.choice(K, size=min_sample_size, replace=False) # get 4 random matches
        
        sample_points1 = points_im1[sample_indices]
        sample_points2 = points_im2[sample_indices]

        curr_h = computeH(sample_points1, sample_points2)

        if curr_h is None:
            continue

        for i in range(K):
            p1 = points_im1[i]
            p1 = np.append(p1, 1)

            p2 = points_im2[i]

            p1_transformed = curr_h @ p1

            w_prime = p1_transformed[2]
            if np.abs(w_prime) > 1e-6: # division by zero averted
                c_pred = p1_transformed[0] / w_prime
                r_pred = p1_transformed[1] / w_prime
                p1_transformed_cartesian = np.array([c_pred, r_pred]) 
            else:
                 # outlier
                 continue 
            
            dist = np.linalg.norm(p1_transformed_cartesian - p2)

            if dist < inlier_threshold:
                curr_inlier_mask[i] = True
                current_inliers += 1
        
        if current_inliers > max_inliers:
            max_inliers = current_inliers
            best_h = curr_h
            best_inlier_mask = curr_inlier_mask

        num += 1

    if best_inlier_mask is not None:
        best_inlier_mask = np.array(best_inlier_mask)

    inlier_points1 = points_im1[best_inlier_mask]
    inlier_points2 = points_im2[best_inlier_mask]

    if len(inlier_points1) > 4:
        H_final = computeH(inlier_points1, inlier_points2)
    else:
        print("failure, not enough inlier points")
        return None

    return H_final

def warp_image(h_final, im1, im2):
    warped_image, corners = warpImageBilinear(im1, h_final)
    final_image = stitch(warped_image, im2, corners)


suppressed_im, points_of_interest = adaptive_non_max_suppression(harris_im, im)

features = feature_descriptor_extraction(im_gray, points_of_interest)
# pano1
# im1 = cv2.imread("./images/im1.JPG")
# im2 = cv2.imread("./images/im2.JPG")

# pano2
im1 = cv2.imread("./images/im8.JPG")
im2 = cv2.imread("./images/im9.JPG")

# im1 = cv2.imread("./images/im8.JPG")
# im2 = cv2.imread("./images/im9.JPG")

match_points_im1, match_points_im2 = feature_matching(im1, im2)

match_points_im1_xy = match_points_im1[:, ::-1] # Swaps columns
match_points_im2_xy = match_points_im2[:, ::-1] # Swaps columns

h_final = ransac(match_points_im1_xy, match_points_im2_xy, 4, 5000, 5)

warp_image(h_final, im1, im2)









# cv2.imshow("image with suppressed harris", suppressed_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# print(harris_corners)