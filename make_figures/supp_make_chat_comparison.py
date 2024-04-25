"""
Compare sanitized images.

"""

import matplotlib.pyplot as plt
import os
from PIL import Image
import os
from utils.utils_make_pictures import pretty_picture, pretty_comparison, pretty_comparison_vertical
    
    
if __name__ == "__main__":
    input_path = "results/rq1"
    output_path = "results/pretty_comparison_vertical.pdf"
    # suds
    pretty_comparison_vertical(input_path, output_path, demo="lsb_demo", imsize=64)
    