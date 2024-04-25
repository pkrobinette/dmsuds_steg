"""
Create a diagram for generated steg images

"""

import matplotlib.pyplot as plt
import os
from PIL import Image
import os

# where pictures are held for each im-type
directories = [
        'C',
        'C_prime',
        'C_hat',
        'C_res_5x',
        'S',
        'S_prime',
        'S_hat']


def pretty_picture_imagenet(input_path, save_name, imsize=64):
    """
    Make a pretty picture
    """
    # Set the dimensions of each individual image
    width, height = imsize, imsize
    
    # Set the number of images per row and column in the final image
    images_per_row = 7
    images_per_column = 4
    padding = 20  # Set the padding value as needed
    ims = 3
    
    combined_image = Image.new("RGB", (width * images_per_row, height * images_per_column + padding*(images_per_column//3)), (255,255,255))
    
    hide_type = ["rq4_ddh", "rq4_pris"]
    
    # Iterate through the images and paste them into the combined image
    for i, h in enumerate(hide_type):
        # each type only gets 2
        for j in range(2):
            for col, d in enumerate(directories):
                p = os.path.join(input_path, h, d)

                if h == "imagenet_pris":
                    image_file = str((i+2)*j +3 ) + ".jpg"
                else:
                    image_file = str(j) + ".jpg"
                
                # print(p, image_file)
                image = Image.open(os.path.join(p, image_file))
                image = image.resize((width, height))
        
                # Calculate the position of the current image
                x = col * width
                # y = ((i*2)+j) * height
                
                if (i*2)+j >= 2 and (i*2)+j < 4:
                    y = ((i*2)+j) * height + padding
                elif (i*2)+j >= 4:
                    y = ((i*2)+j) * height + 2 * padding
                else:
                    y = ((i*2)+j) * height

        
                # Paste the current image into the combined image
                combined_image.paste(image, (x, y))
    
    # Save the combined image
    # save_path = input_path + save_name if input_path[-1] == "/" else input_path + "/" + save_name
    combined_image.save(save_name)
    
    print("Saved to: ", save_name)

def pretty_picture(input_path, save_name, imsize=64, dataset="mnist"):
    """
    Make a pretty picture
    """
    # Set the dimensions of each individual image
    width, height = imsize, imsize
    
    # Set the number of images per row and column in the final image
    images_per_row = 7
    images_per_column = 6
    padding = 20  # Set the padding value as needed
    ims = 3
    
    # Create a blank image with the combined dimensions
    imtype = "L" if dataset == "mnist" else "RGB"
    color = 255 if dataset == "mnist" else (255, 255, 255)
    combined_image = Image.new(imtype, (width * images_per_row, height * images_per_column + padding*(images_per_column//3)), color)
    
    hide_type = ["lsb_demo", "ddh_demo", "udh_demo"]
    
    # Iterate through the images and paste them into the combined image
    for i, h in enumerate(hide_type):
        # each type only gets 2
        for j in range(2):
            for col, d in enumerate(directories):
                p = os.path.join(input_path, h, d)

                image_file = str((i*2)+j+7) + ".jpg"
                
                # print(p, image_file)
                image = Image.open(os.path.join(p, image_file))
                image = image.resize((width, height))
        
                # Calculate the position of the current image
                x = col * width
                # y = ((i*2)+j) * height
                
                if (i*2)+j >= 2 and (i*2)+j < 4:
                    y = ((i*2)+j) * height + padding
                elif (i*2)+j >= 4:
                    y = ((i*2)+j) * height + 2 * padding
                else:
                    y = ((i*2)+j) * height

        
                # Paste the current image into the combined image
                combined_image.paste(image, (x, y))
    
    # Save the combined image
    # save_path = input_path + save_name if input_path[-1] == "/" else input_path + "/" + save_name
    combined_image.save(save_name)
    
    print("Saved to: ", save_name)

def pretty_comparison(input_path, save_name, demo="lsb_demo", imsize=64, dataset="cifar"):
    """
    Make a pretty picture
    """
    # Set the dimensions of each individual image
    width, height = imsize, imsize
    
    # Set the number of images per row and column in the final image
    images_per_row = 12
    images_per_column = 3
    padding = 0  # Set the padding value as needed
    
    # Create a blank image with the combined dimensions
    imtype = "L" if dataset == "mnist" else "RGB"
    color = 255 if dataset == "mnist" else (255, 255, 255)
    combined_image = Image.new(imtype, (width * images_per_row, height * images_per_column + padding*(images_per_column//3)), color)
    
    sani_type = [f"suds/{demo}/C_prime", f"suds/{demo}/C_hat", f"diffusion_model/{demo}/C_hat"]
    
    # Iterate through the images and paste them into the combined image
    for row, h in enumerate(sani_type):
        for col in range(images_per_row):
            p = os.path.join(input_path, h)

            image_file = str(col) + ".jpg"
            image = Image.open(os.path.join(p, image_file))
            image = image.resize((width, height))
    
            # Calculate the position of the current image
            x = col * width
            y = row * (height + padding)
    
            # Paste the current image into the combined image
            combined_image.paste(image, (x, y))
    
    # Save the combined image
    combined_image.save(save_name)
    
    print("Saved to: ", save_name)

from PIL import Image
import os

def pretty_comparison_vertical(input_path, save_name, demo="lsb_demo", imsize=64):
    """
    Make a pretty picture with images from specific directories
    """
    # Set the dimensions of each individual image
    width, height = imsize, imsize

    directories = [f"noise/{demo}/C", f"noise/{demo}/C_hat", f"suds/{demo}/C_hat", f"diffusion_model/{demo}/C_hat"]
    
    # Set the number of images per row and column in the final image
    images_per_row = len(directories)
    images_per_column = 8  # Assuming you want 10 images per column
    padding = 5  # Set the padding value as needed
    
    # Create a blank image with the combined dimensions
    color = (255, 255, 255)
    combined_image = Image.new("RGB", (width * images_per_row + padding * (images_per_row - 1), 
                                      height * images_per_column + padding * (images_per_column - 1)), color)
    
    # Iterate through the directories and images and paste them into the combined image
    for col, directory in enumerate(directories):
        for row in range(images_per_column):
            image_file = os.path.join(input_path, directory, f"{row + 10}.jpg")  # Adjust as per the naming convention of your files
            if not os.path.exists(image_file):
                print(f"Image {image_file} not found.")
                continue

            image = Image.open(image_file)
            image = image.resize((width, height))
    
            # Calculate the position of the current image
            x = col * (width + padding)
            y = row * (height + padding)
    
            # Paste the current image into the combined image
            combined_image.paste(image, (x, y))
    
    # Save the combined image
    combined_image.save(save_name)
    
    print("Saved to: ", save_name)
