import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib as plt
from tqdm import tqdm
import math
import imageio


def set_seed(seed):
    # to get repeatable results you can choose to set seed, i like this so that the model doesnt auto predict zero weights and get cooked
    random.seed(seed)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # makes cuda deterministic 
    torch.backends.cudnn.deterministic = True
    
    # This disables the cuDNN auto-tuner, which can select 
    # different algorithms on different runs
    torch.backends.cudnn.benchmark = False

set_seed(180)

device = "mps" # change if you are going to run this on colab like i did
data = np.load(f"./data/lego_200x200.npz")
h, w = data["images_train"].shape[1:3]

torch.manual_seed(180)

# Training images: [100, 200, 200, 3]
images_train = torch.tensor(data["images_train"] / 255.0, device=device, dtype=torch.float32)

# Cameras for the training images 
# (camera-to-world transformation matrix): [100, 4, 4]
c2ws_train = torch.tensor(data["c2ws_train"], device=device, dtype=torch.float32)

# Validation images: 
images_val = torch.tensor(data["images_val"] / 255.0, device=device, dtype=torch.float32)

# Cameras for the validation images: [10, 4, 4]
c2ws_val = torch.tensor(data["c2ws_val"],  device=device, dtype=torch.float32)

# Test cameras for novel-view video rendering: 
# (camera-to-world transformation matrix): [60, 4, 4]
c2ws_test = torch.tensor(data["c2ws_test"], device=device, dtype=torch.float32)

# Camera focal length
focal = torch.tensor(data["focal"], device=device, dtype=torch.float32)  # float

K = torch.tensor([
    [focal, 0, w / 2.0],
    [0, focal, h / 2.0],
    [0, 0, 1]
], device=device, dtype=torch.float32)

K_inv = torch.linalg.inv(K)


class DataLoader:
    def __init__(self, images, K, c2w):
        self.images = images
        self.K = K
        self.c2w = c2w
        self.number_images, self.h, self.w, _ = images.shape
        self.number_pixels_img = self.h * self.w

        # creating our grid of coords
        uu = torch.arange(0, self.w, device=device).float()
        vv = torch.arange(0, self.h, device=device).float()
        u, v = torch.meshgrid(uu, vv, indexing='xy')
        uv_grid = torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)

        # copy this grid for every image
        uv_tiled = uv_grid.repeat(self.number_images, 1)

        # copy the c2w matrix for every pixel
        c2w_repeated = self.c2w.repeat_interleave(self.number_pixels_img, dim=0)
    
        # call pixel to ray function with our batched data
        self.rays_o, self.rays_d = pixel_to_ray(self.K, c2w_repeated, uv_tiled)
        self.pixels = self.images.view(-1, 3)
    
    def sample_rays(self, batch_size):
        index = torch.randint(0, self.pixels.shape[0], (batch_size,), device=device)
        # return the precomped rays/pixels
        return self.rays_o[index], self.rays_d[index], self.pixels[index]



def camera_to_world(c2w, x_c): # hypothetically x_c is a batch of points for a certain camera
    ones = torch.ones((x_c.shape[0], 1), device=device) # get the number of points. so if we have n x 3. 
    # we have an n x 1 column of 1's
    x_c = torch.concat([x_c, ones], axis=1) # all the points now have a 1 
    x_c = x_c.unsqueeze(-1)
    
    result = c2w @ x_c # n x 4 x 4 @ (n x 4 x 1) which is n operations of 4 x 4 @ 4 x 1
    result = result[:, :3] / result[:, 3:4] # dividing by the last component
    
    return result

#transforming 2d pixel coordinate from an image into a 3d position relative to the camera
def pixel_to_camera(K, uv, s): 
    ones = torch.ones((uv.shape[0], 1), device=device)
    centered_uv = uv + 0.5  # center of the pixel 
    uv = torch.concat([centered_uv, ones], axis=1) 

    result = (s * K_inv @ uv.t()).t()
    return result

def pixel_to_ray(K, c2w, uv):
    ray_origin = c2w[:, :3, 3] # all rows, third column

    camera_coord = pixel_to_camera(K, uv, 1)
    world_coord = camera_to_world(c2w, camera_coord).squeeze(-1) # get (5, 3, 1) into (5, 3)
    
    ray_direction = (world_coord - ray_origin) / (torch.norm(world_coord - ray_origin, dim=-1, keepdim=True))
    return ray_origin, ray_direction

image = images_train[0]
h, w, _ = image.shape

t_width = 4 / 64


def sample_along_rays(r_o, r_d, perturb=True, near=2, far=6, n_samples=64):
    t = torch.linspace(near, far, n_samples, device=device)
    if perturb:
        t = t + torch.rand_like(t) * (far - near) / n_samples
    # unsqueeze and repeat to get in a usable shape
    r_o_expanded = r_o.unsqueeze(1).repeat(1, n_samples, 1)
    r_d_expanded = r_d.unsqueeze(1).repeat(1, n_samples, 1)

    t = t.reshape(1, n_samples, 1)
    return r_o_expanded + r_d_expanded * t

import viser, time  # pip install viser
# import numpy as np


H, W = images_train.shape[1:3]

# server = viser.ViserServer(share=True)

combined_train = torch.cat([images_train, images_val], dim=0)
combined_c2ws = torch.cat([c2ws_train, c2ws_val],dim=0)
dataset = DataLoader(combined_train, K, combined_c2ws)
print(len(images_train))

rays_o, rays_d, pixels = dataset.sample_rays(100)

points = sample_along_rays(rays_o, rays_d, perturb=True)

# for i, (image, c2w) in enumerate(zip(combined_train, combined_c2ws)):
  
#   image_np = image.detach().cpu().numpy()
#   c2w_np = c2w.detach().cpu().numpy()
#   K_focal_np = K[0, 0].detach().cpu().numpy() 
  
#   server.scene.add_camera_frustum(
#     f"/cameras/{i}",
#     fov=2 * np.arctan2(H / 2, K_focal_np), 
#     aspect=W / H,
#     scale=0.02, # need to adjust if you're using my data or lego data. my data is too tight on scale =0.15
#     wxyz=viser.transforms.SO3.from_matrix(c2w_np[:3, :3]).wxyz,
#     position=c2w_np[:3, 3],
#     image=image_np
#   )
#   rays_o, rays_d, pixels = dataset.sample_rays(100)
#   points = sample_along_rays(rays_o, rays_d, perturb=True)
#   ray_data = {"rays_o": rays_o, "rays_d": rays_d}

#   for i, (o, d) in enumerate(zip(ray_data["rays_o"], ray_data["rays_d"])):
#     o_np = o.detach().cpu().numpy()
#     d_np = d.detach().cpu().numpy()

#     positions = np.stack((o_np, o_np + d_np * 6.0))
#     server.add_spline_catmull_rom(
#         f"/rays/{i}", positions=positions,
#     )
#     points_np = points.detach().cpu().numpy() 

#     server.add_point_cloud(
#         f"/samples",
#         colors=np.zeros_like(points_np).reshape(-1, 3), 
#         points=points_np.reshape(-1, 3),
#         point_size=0.03,
#     )
# while True:
#     time.sleep(0.1)
    

def positional_encoding(coords, positional_frequency):
    original_coords = coords
    D = original_coords.shape[-1] 
    batch_shape = original_coords.shape[:-1] 
    
    freqs = (2 ** torch.arange(start=0, end=positional_frequency, device=device)) * 2 * torch.pi

    scaled_coords = original_coords.unsqueeze(-1) * freqs

    sines = torch.sin(scaled_coords)
    cosines = torch.cos(scaled_coords)
    
    trig_part = torch.cat([sines, cosines], dim=-1)

    encoded_trig = trig_part.reshape(*batch_shape, D * 2 * positional_frequency)
    
    encoded_data = torch.cat([original_coords, encoded_trig], dim=-1)

    return encoded_data


class NeRF(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(63, 256)
        self.linear_2 = nn.Linear(256, 256)
        self.linear_3 = nn.Linear(256, 256)
        self.linear_4 = nn.Linear(256, 256)

        self.linear_5 = nn.Linear(319, 256) # this guy will take in 256 + width of x which is 63
        self.linear_6 = nn.Linear(256, 256) 
        self.linear_7 = nn.Linear(256, 256)
        self.linear_8 = nn.Linear(256, 256)

        self.linear_density = nn.Linear(256, 1) # diverts to density

        self.linear_9 = nn.Linear(256, 256) # continues onto rgb
        self.linear_10 = nn.Linear(283, 128) # here we will have 256 + rd size which is 27

        self.linear_rgb = nn.Linear(128, 3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_encoded, r_d):
        x = self.relu(self.linear_1(x_encoded))
        x = self.relu(self.linear_2(x))
        x = self.relu(self.linear_3(x))
        x = self.relu(self.linear_4(x))

        x = torch.concat([x, x_encoded], dim=-1)

        x = self.relu(self.linear_5(x))
        x = self.relu(self.linear_6(x))
        x = self.relu(self.linear_7(x))
        x = self.linear_8(x)
        # print("output before density layer ", output_7[0:5])
        
        density = self.relu(self.linear_density(x))

        x = self.linear_9(x)
        x = torch.concat([x, r_d], dim=-1)

        x = self.relu(self.linear_10(x))
        color = self.linear_rgb(x)
        sig_color = self.sigmoid(color)

        # print("density ", density)

        return density, sig_color


def volrend(sigmas, rgbs, step_size):
    deltas = torch.full_like(sigmas, step_size)
    
    alphas = 1.0 - torch.exp(-sigmas * deltas)
    
    # transmittances
    one_minus_alphas = 1.0 - alphas

    # pad with a 1 so that cumprod works
    T_pad = torch.ones_like(one_minus_alphas[:, :1]) 
    T_unrolled = torch.cat([T_pad, one_minus_alphas], dim=1) 
    
    T_cumprod = torch.cumprod(T_unrolled, dim=1)
    
    # slice last useless elem
    T = T_cumprod[:, :-1] # [N_rays, N_samples, 1]

    # final weights/colors
    weights = T * alphas 
    
    # final colors C is weights times the colors
    final_color = torch.sum(weights * rgbs, dim=1) # [N_rays, 3]
    
    return final_color

step_size = (6.0 - 2.0) / 64

steps = 10000
batch_size = 10000

model = NeRF().to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

def render_full_image(model, K, c2w, h, w, n_samples_render=64, batch_size=10000):
    model.eval()
    
    near, far = 2.0, 6.0
    step_size = (far - near) / n_samples_render

    uu = torch.arange(0, w, device=device).float()
    vv = torch.arange(0, h, device=device).float() 
    u, v = torch.meshgrid(uu, vv, indexing='xy')
    
    uv = torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)

    all_pixels = []

    with torch.no_grad():  
        for i in range(0, h * w, batch_size): # batched to prevent mem explosion
            uv_batch = uv[i:i + batch_size]

            c2w_batch = c2w.unsqueeze(0).repeat(len(uv_batch), 1, 1)
            rays_o, rays_d = pixel_to_ray(K, c2w_batch, uv_batch)

            t = torch.linspace(near, far, n_samples_render, device=device)
            t = t.repeat(len(uv_batch), 1)
            
            r_o_expanded = rays_o.unsqueeze(1).repeat(1, n_samples_render, 1)
            r_d_expanded = rays_d.unsqueeze(1).repeat(1, n_samples_render, 1)
            points = r_o_expanded + r_d_expanded * t.unsqueeze(-1)
            
            encoded_points = positional_encoding(points, 10)
            encoded_rd = positional_encoding(r_d_expanded, 4)
            sigmas, colors = model(encoded_points, encoded_rd)

            predicted_pixels = volrend(sigmas, colors, step_size)
            
            all_pixels.append(predicted_pixels)

    
    model.train()
    
    return torch.cat(all_pixels, dim=0).reshape(h, w, 3)

# to create lego gif
# def final_test_renders(model, test_dataset):
#   model.eval()

#   with torch.no_grad():
#     for i in range(test_dataset.c2w.shape[0]):
#       image = render_full_image(model, K, test_dataset.c2w[i], h, w, 64, 10000)
#       image_np = image.detach().cpu().numpy()
#       plt.imsave(f"final_lego_gif_images/image_{i}.JPG", image_np)
cam_origins = c2ws_train[:, :3, 3]
distances = torch.norm(cam_origins, dim=-1)

# finding camera positions to update step size and near/far.
min_dist = torch.min(distances)
max_dist = torch.max(distances)

train_dataset = DataLoader(images_train, K, c2ws_train)
val_dataset = DataLoader(images_val, K, c2ws_val)
test_dataset = DataLoader(images_train[:60], K, c2ws_test) # need to change the amount of images if working with my set,
# will cause an error if not

val_step_list = []
val_psnr_list = []
PSNR_list = []
import cv2
for step in range(2000):
    rays_o, rays_d, pixels = train_dataset.sample_rays(batch_size)

    points = sample_along_rays(rays_o, rays_d, perturb=True)
    # print("points shape ", points.shape)
    # rays_o, rays_d, points, img_idx, rows, cols = sample_along_rays(batch_size, K, c2ws_train, h, w) # points is the data we need here
    # print("c2ws training ", c2ws_train[0:5])
    # print("rays_o", rays_o[0:5], "rays_d ", rays_d[0:5], "points ", points[0:5])

    H, W = images_train.shape[1:3]

    # true_colors = images_train[img_idx, rows.long(), cols.long()]

    encoded_points = positional_encoding(points, 10)
    # print("ray_d shape ", rays_d.shape)
    encoded_rays_d = positional_encoding(rays_d, 4).unsqueeze(1).repeat(1, 64,1) # 64 points on the ray get the same direction vector

    optimizer.zero_grad()

    sigmas, colors = model(encoded_points, encoded_rays_d)

    predicted_colors = volrend(sigmas, colors, step_size)


    loss = loss_fn(predicted_colors, pixels)
    loss.backward()
    optimizer.step()

    PSNR = 10 * torch.log10(1 / loss)
    PSNR_list.append(PSNR.detach().cpu().numpy())
    

    if step % 10 == 0:
        print("loss ", loss, " PSNR ", PSNR)
    
    
    if step % 10 == 0:
        model.eval()
        with torch.no_grad(): # test on one image 
            im = render_full_image(model, K, c2ws_val[0], h, w, 64, 10000)
            
            gt_image = images_val[0]
            
            val_loss = loss_fn(im, gt_image)
            
            val_psnr = 10 * torch.log10(1 / val_loss)
            val_psnr_list.append(val_psnr.detach().cpu().numpy())
            val_step_list.append(step)

            print(f"step {step}")
            print(f"val Loss {val_loss.item():.6f}")
            print(f"val PSNR {val_psnr.item():.4f}")
            
        model.train()

plt.plot(val_step_list, val_psnr_list)
plt.title("PSNR validation Curve for lego image")
plt.xlabel("Epoch")
plt.ylabel("PSNR value")
plt.show()

# plt.plot(range(0, 3000), PSNR_list)
# plt.title("PSNR Training Curve for Personal Image")
# plt.xlabel("Epoch")
# plt.ylabel("PSNR value")
# plt.show()

# testing
model = NeRF().to(device)
model.load_state_dict(torch.load("weights/my_model.pth"))
model.eval()


def look_at_origin(pos):  # cam towards orig
    forward = -pos / np.linalg.norm(pos)  # normalize
    up_world = np.array([0, 1, 0])
    
    # compute right vec
    right = np.cross(forward, up_world) 
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward) 
    
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    return c2w


model.eval()

# get a start pos from c2ws
start_pos = c2ws_train[0, :3, 3].cpu().numpy()
frames = []

with torch.no_grad():
    for phi in tqdm(np.linspace(0., 360., 60, endpoint=False)):
        
        # new cam pos
        phi_rad = phi / 180. * np.pi
        
        pos = np.array([0.25 * math.cos(phi_rad), 0.0, 0.25 * math.sin(phi_rad)])
        
        c2w_np = look_at_origin(pos)
        
        c2w_tensor = torch.tensor(c2w_np, dtype=torch.float32).to(device)
        img_tensor = render_full_image(model, K, c2w_tensor, h, w, n_samples_render=64, batch_size=10000
        )
        
        frame_np = img_tensor.detach().cpu().numpy()
        frame = (frame_np * 255).astype(np.uint8)
        
        frames.append(frame)

imageio.mimsave("my_nerf_render.gif", frames, fps=10, loop=0)









