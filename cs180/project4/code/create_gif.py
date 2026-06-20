

import imageio
import glob
import re
import os
import numpy as np

# get them sorted in the right order so the gif makes sense
def create_gif(image_folder, gif_path, fps=10):
    def sort_key(filename):
        number = re.search(r"(\d+)", os.path.basename(filename))
        return int(number.group(1)) if number else 0

    # get im files
    file_paths = glob.glob(f"{image_folder}/*.jpg")
    file_paths.sort(key=sort_key)
    
    if not file_paths:
        print("FUCK")
        return

    images = []
    for file_path in file_paths:
        im = imageio.imread(file_path)
        im = np.rot90(im, 2)
        images.append(im)
    
    imageio.mimsave(gif_path, images, fps=fps, loop=0)

create_gif("./my_render_frames", "./my_nerf_results/personal_gif.gif", fps=20)