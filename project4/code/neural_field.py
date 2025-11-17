import numpy as np
import cv2
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
device = 'mps'



model = nn.Sequential(
    nn.Linear(in_features=42, out_features=256), # use a frequency level of 10 to get a positional encoding of your x, y
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=3),
    nn.Sigmoid()
)
model.to(device) # send the model to the gpu

#./golden_gate.JPG for my image of choice. fox.jpg for the provided image.
image = cv2.imread("./mlp_images/golden_gate.JPG")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
normalized_image = image_rgb.astype(np.float32) / 255.0
image_tensor = torch.tensor(normalized_image, device=device)

h, w, _ = image.shape

epochs = 3000
batch_size = 10000

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# creating grids of coordinates
X_coords = torch.arange(start=0, end=w, device=device) 
y_coords = torch.arange(start=0, end=h, device=device)
yv, xv = torch.meshgrid(y_coords, X_coords)
# print("shape", yv.shape)

all_coords = torch.stack([yv, xv], axis=2).reshape(-1, 2)


def positional_encoding(coords, batch_size, positional_frequency):
    freqs = (2 ** torch.arange(start=0, end=positional_frequency, device=device)) * torch.pi # (pi, 2pi, 4pi etc)
    scaled_coords = coords.reshape(batch_size, 2, 1) * freqs.reshape(1, 1, positional_frequency)
    scaled_coords = scaled_coords.flatten(1)
    
    sine_coords = torch.sin(scaled_coords)
    cosine_coords = torch.cos(scaled_coords)

    encoded_trig = torch.cat([sine_coords, cosine_coords], dim=1) # 4096, 40
    encoded_data = torch.cat([coords, encoded_trig], dim=1) # 4096, 42

    return encoded_data

PSNR_list = []
positional_frequency = 10
for epoch in range(epochs + 1):
    # get data 
    rand_rows = torch.randint(0, h, size=(batch_size,), device=device) # (4096)
    rand_cols = torch.randint(0, w, size=(batch_size,), device=device)

    y = image_tensor[rand_rows, rand_cols]

    rand_rows = rand_rows.float() / (h - 1)
    rand_cols = rand_cols.float() / (w - 1)
    coords = torch.cat([rand_rows.unsqueeze(1), rand_cols.unsqueeze(1)], dim=1) # 4096, 2

    encoded_data = positional_encoding(coords, batch_size, positional_frequency)

    y_pred = model(encoded_data)

    loss = loss_fn(y_pred, y)
    PSNR = 10 * torch.log10(1 / loss)

    PSNR_list.append(PSNR.detach().cpu().numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("loss: ", loss, "PSNR: ", PSNR)

    if epoch % 500 == 0:
        all_coords_normalized = all_coords.float() / torch.tensor([h - 1, w - 1], device=device)
        encoded_data = positional_encoding(all_coords_normalized, h * w, positional_frequency)
        Y_pred = model(encoded_data)
        # print("shape of y_pred: ", Y_pred.shape)
        Y_pred = Y_pred.reshape((h, w, 3))
        # print("shape of y_pred: ", Y_pred.shape)

        Y_pred_numpy = Y_pred.detach().cpu().numpy()

        Y_pred_img = np.clip(Y_pred_numpy * 255.0, 0, 255).astype(np.uint8)

        Y_pred_bgr = cv2.cvtColor(Y_pred_img, cv2.COLOR_RGB2BGR)

        cv2.imshow(f"Epoch: {epoch}", Y_pred_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# plotting psnr
plt.plot(range(0, 3000), PSNR_list)
plt.title("PSNR Curve for Golden Gate")
plt.xlabel("Epoch")
plt.ylabel("PSNR value")
plt.show()







    