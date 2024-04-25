"""
Create a diagram for generated steg images.


Note: utils must be added to path.
"""

import matplotlib.pyplot as plt
import os
from PIL import Image
import os
from utils.utils_make_pictures import pretty_picture, pretty_comparison
    
    
if __name__ == "__main__":
    input_path = "../results/rq1/suds"
    output_path = "../results/rq1_suds.pdf"
    # suds
    pretty_picture(input_path, output_path, dataset="cifar")
    # diffusion model
    input_path = "../results/rq1/diffusion_model"
    output_path = "../results/rq1_dmsuds.pdf"
    pretty_picture(input_path, output_path, dataset="cifar")
    # noise
    input_path = "../results/rq1/noise"
    output_path = "../results/rq1_noise.pdf"
    pretty_picture(input_path, output_path, dataset="cifar")
    # dct_noise
    input_path = "../results/rq1/dct_noise"
    output_path = "../results/rq1_dct_noise.pdf"
    pretty_picture(input_path, output_path, dataset="cifar")
    