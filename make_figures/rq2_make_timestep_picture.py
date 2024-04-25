"""
Create a diagram for comparing different timesteps of diffusion model
sanitization.

"""

import matplotlib.pyplot as plt
import os
from PIL import Image
import os

def timestep_picture(input_path, save_name, ref_image_path, img_num=0, padding=10, imsize=64, dataset="mnist"):
    """
    Make a pretty picture
    """
    # Set the dimensions of each individual image
    width, height = imsize, imsize
    
    # Fetch all step directories and sort them
    step_dirs = [d for d in os.listdir(input_path) if d.startswith('step_')]
    step_dirs = sorted(step_dirs, key=lambda x: int(x.split('_')[1]))[:12]

    # Number of step directories determines columns
    num_cols = 2 + len(step_dirs)  # +1 for the reference image
    
    # Two rows for C_hat and S_hat
    num_rows = 2
    
    # Create a blank image with the combined dimensions (plus padding for reference image)
    imtype = "L" if dataset == "mnist" else "RGB"
    color = 255 if dataset == "mnist" else (255, 255, 255)
    combined_width = width * num_cols + padding
    combined_image = Image.new(imtype, (combined_width, height * num_rows), color)

    # Add the reference image to both rows
    ref_image_cover = Image.open(os.path.join(ref_image_path, "C", f"{img_num}.jpg")).resize((width, height))
    ref_image_secret = Image.open(os.path.join(ref_image_path, "S", f"{img_num}.jpg")).resize((width, height))
    combined_image.paste(ref_image_cover, (0, 0))
    combined_image.paste(ref_image_secret, (0, height))

    for col, step_dir in enumerate(step_dirs, start=1):  # start=1 to account for reference image
        # Fetch image from C_hat directory
        c_hat_image_path = os.path.join(input_path, step_dir, 'C_hat', f'{img_num}.jpg')
        c_hat_image = Image.open(c_hat_image_path).resize((width, height))
        combined_image.paste(c_hat_image, ((col * width), 0))  # Pasting on the first row with padding
        
        # Fetch image from S_hat directory
        s_hat_image_path = os.path.join(input_path, step_dir, 'S_hat', f'{img_num}.jpg')
        s_hat_image = Image.open(s_hat_image_path).resize((width, height))
        combined_image.paste(s_hat_image, ((col * width), height))  # Pasting on the second row with padding

    # After placing all the timesteps, add the reference images to the end with padding
    fin_step_path_c = os.path.join(input_path, "step_1000", 'C_hat', f'{img_num}.jpg')
    fin_step_path_s = os.path.join(input_path, "step_1000", 'S_hat', f'{img_num}.jpg')
    ref_image_cover = Image.open(fin_step_path_c).resize((width, height))
    ref_image_secret = Image.open(fin_step_path_s).resize((width, height))
    combined_image.paste(ref_image_cover, (width * (num_cols-1) + padding, 0))
    combined_image.paste(ref_image_secret, (width * (num_cols-1) + padding, height))


    # Save the combined image
    combined_image.save(save_name)
    
    print("Saved to: ", save_name)
    
if __name__ == "__main__":
    img_num = 0
    for steg in ["lsb", "ddh", "udh"]:
        input_path = f"../results/rq2/{steg}"
        output_path = f"../results/rq2_timesteps_{steg}_{img_num}.pdf"
        ref_path = "../results/rq2"
        timestep_picture(input_path, output_path, ref_path, img_num=img_num, dataset="cifar")
    