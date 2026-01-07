# calibrating the camera
import cv2
import numpy as np
import sklearn
import sys

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

tag_point = np.array([[0.0, 0.0,  0.0], [0.05, 0.0, 0.0],
    [0.05, 0.05, 0.0], [0.0,  0.05, 0.0]
])

world_points = []
image_points = []

for i in range(0, 45):
    image = cv2.imread(f"./resized_calibration_images/im{i}.JPG", cv2.IMREAD_GRAYSCALE)    
    corners, ids, _ =  detector.detectMarkers(image)

    if ids is not None:
        for corner in corners:
            world_points.append(tag_point)
            image_points.append(corner)

world_points = np.array(world_points, dtype=np.float32)
image_points = np.array(image_points, dtype=np.float32)
h, w = image.shape
print("world points, ", world_points)
ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints=world_points, imagePoints=image_points, imageSize=(w, h), cameraMatrix=None, distCoeffs=None)

print("intrinsics: ", intrinsic_mtx) # save this so we don't have to recalibrate every time we run our program
print("dist,", dist)

# intrinsic_mtx = np.array([[201.1440429, 0, 138.50602483],
#  [0, 200.6027163, 98.49778947],
#  [0, 0, 1]])

# dist = np.array([[ 0.24238198, -0.54468397, -0.02576084, -0.01218189, 0.08893795]])


c2ws = []
images = []


# go through our new images, detect the SINGLE aruco tag. so there should only be 1 corners list
for i in range(0, 95):
    if i == 20:
        continue
    world_points = []
    image_points = []

    image = cv2.imread(f"./final_data/im{i}.JPG")
    corners, ids, _ =  detector.detectMarkers(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if ids is not None:
        for corner in corners:
            world_points.append(tag_point)
            image_points.append(corner)
    else:
        continue # if its none, we move on to the next image

    world_points = np.array(world_points).reshape(-1, 3) # just making sure its (N, 3)
    image_points = np.array(image_points).reshape(-1, 2)

    ret, rvec, tvec = cv2.solvePnP(world_points, image_points, intrinsic_mtx, dist)
    if ret: # inverting the matrix to go from world-to-cameara pose to a camera-to-world pose.
        images.append(image_rgb)
        R, _ = cv2.Rodrigues(rvec) 
        
        w2c_matrix = np.eye(4, dtype=np.float32)
        w2c_matrix[:3, :3] = R
        w2c_matrix[:3, 3] = tvec.squeeze() # tvec is (3,1), make it (3,)

        c2w_matrix = np.linalg.inv(w2c_matrix)
        c2w_matrix[:3, 3] *= 10

        c2ws.append(c2w_matrix)

images = np.array(images)
# print("LENGTH OF IMAGES, ", len(images))
c2ws = np.array(c2ws)
# print("images shape", images.shape, "c2ws shape ", c2ws.shape)

h, w = images[0].shape[:2]



# undistort and crop
# --- UNDISTORTION (PRESERVING 100% OF PIXELS) ---

# alpha=1 ensures ALL original pixels are retained. 
# It adds black pixels at the corners instead of cutting off the image.
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    intrinsic_mtx, dist, (w, h), alpha=1, newImgSize=(w, h)
)

# We do NOT use the ROI to crop. We want the full image.
images_undistorted = []

print("Undistorting images...")
for img in images:
    # This function uses the new matrix to remap the pixels correctly
    undistorted_img = cv2.undistort(img, intrinsic_mtx, dist, None, new_camera_matrix)
    
    # NO CROPPING HERE. Just append the full result.
    images_undistorted.append(undistorted_img)

images = np.array(images_undistorted)

# IMPORTANT: Your K matrix has changed because the image center shifted.
# Use the new matrix for your NeRF training.
K_final = new_camera_matrix

n_total = len(images)

# create train/val/test splits
train_split_idx = int(n_total * 0.7)
val_split_idx = train_split_idx + int(n_total * 0.15)

images_train = images[:train_split_idx]
c2ws_train = c2ws[:train_split_idx]

images_val = images[train_split_idx:val_split_idx]
c2ws_val = c2ws[train_split_idx:val_split_idx]

images_test = images[val_split_idx:]
c2ws_test = c2ws[val_split_idx:]

print(images_train.shape, c2ws_train.shape, images_val.shape, c2ws_val.shape, c2ws_test.shape)
np.savez(
    './data/my_data.npz',
    images_train=images_train,
    c2ws_train=c2ws_train,
    images_val=images_val,
    c2ws_val=c2ws_val,
    c2ws_test=c2ws_test,
    K_matrix=K_final # extract the focal length from the intrinsic matrix fx + fy / 2
)

    


    

    
    





    



