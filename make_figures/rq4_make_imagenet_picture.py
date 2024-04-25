"""
Create a diagram for generated steg images with the ImageNet dataset.

"""

import matplotlib.pyplot as plt
import os
from PIL import Image
import os
import random
from utils.utils_make_pictures import pretty_picture_imagenet

random.seed(30)

#14

# where pictures are held for each im-type
directories = [
        'C',
        'C_prime',
        'C_hat',
        'S',
        'S_prime',
        'S_hat']
    
    
if __name__ == "__main__":
    input_path = "results"
    output_path = "results/rq4_imagenet_ddh_pris.pdf"
    # suds
    pretty_picture_imagenet(input_path, output_path, imsize=128)
    