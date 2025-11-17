

import imageio
import glob
import re
import os

# get them sorted in the right order so the gif makes sense
def create_gif(image_folder, gif_path, fps=10):
    def sort_key(filename):
        number = re.search(r"(\d+)", os.path.basename(filename))
        return int(number.group(1)) if number else 0

    # get im files
    file_paths = glob.glob(f"{image_folder}/*.JPG")
    file_paths.sort(key=sort_key)
    
    if not file_paths:
        return

    images = []
    for file_path in file_paths:
        images.append(imageio.imread(file_path))
    
    imageio.mimsave(gif_path, images, fps=fps, loop=0)

create_gif("./final_lego_gif_images", "lego_gif.gif", fps=10)