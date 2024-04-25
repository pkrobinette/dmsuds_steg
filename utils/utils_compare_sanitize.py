import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from skimage import exposure
import torch

# save folders
C_folder = "C"
Cprime_folder = "C_prime"
Chat_folder = "C_hat"
Cres_folder = "C_res"
Cres5_folder = "C_res_5x"
Cres10_folder = "C_res_10x"
Cres20_folder = "C_res_20x"
S_folder = "S"
Shat_folder = "S_hat"
Sprime_folder = "S_prime"

def _make_folder(path):
    """
    Creates a folder at path if one does not already exist.
    
    Parameters
    ---------
    path : str
        path of intended folder
    """
    os.makedirs(path, exist_ok=True)
    
def _make_image_folders(save_path):
    """
    Make image folders for steganography image saves. 
    """
    #
    # make save directories
    #
    _make_folder(save_path)
    _make_folder(os.path.join(save_path, C_folder))
    _make_folder(os.path.join(save_path, Cprime_folder))
    _make_folder(os.path.join(save_path, Chat_folder))
    _make_folder(os.path.join(save_path, Cres_folder))
    _make_folder(os.path.join(save_path, Cres5_folder))
    _make_folder(os.path.join(save_path, Cres10_folder))
    _make_folder(os.path.join(save_path, Cres20_folder))
    _make_folder(os.path.join(save_path, S_folder))
    _make_folder(os.path.join(save_path, Shat_folder))
    _make_folder(os.path.join(save_path, Sprime_folder))

def _make_steg_folders(save_path):
    _make_folder(save_path)
    _make_folder(os.path.join(save_path, C_folder))
    _make_folder(os.path.join(save_path, Cprime_folder))
    _make_folder(os.path.join(save_path, S_folder))
    _make_folder(os.path.join(save_path, Sprime_folder))

def _make_timestep_folders(save_path):
    _make_folder(save_path)
    _make_folder(os.path.join(save_path, Chat_folder))
    _make_folder(os.path.join(save_path, Shat_folder))

def save_steg_imgs(c, s, cp, sp, save_path):
    print(f"\nSaving Images to {save_path} ... ")
    
    _make_steg_folders(save_path)
    save_images(c, os.path.join(save_path, C_folder))
    save_images(s, os.path.join(save_path, S_folder))
    save_images(cp, os.path.join(save_path, Cprime_folder))    
    save_images(sp, os.path.join(save_path, Sprime_folder))
    

def save_eval_imgs(c, s, cp, sp, ch, sh, cres, cres5, cres10, cres20, save_path):
    """
    Save steganography images to their corresponding folders.
    """
    print(f"\nSaving Images to {save_path} ... ")
    
    _make_image_folders(save_path)
    save_images(c, os.path.join(save_path, C_folder))
    save_images(s, os.path.join(save_path, S_folder))
    save_images(cp, os.path.join(save_path, Cprime_folder))    
    save_images(sp, os.path.join(save_path, Sprime_folder))
    save_images(ch, os.path.join(save_path, Chat_folder))
    save_images(sh, os.path.join(save_path, Shat_folder))
    save_images(cres, os.path.join(save_path, Cres_folder))
    save_images(cres5, os.path.join(save_path, Cres5_folder))
    save_images(cres10, os.path.join(save_path, Cres10_folder))
    save_images(cres20, os.path.join(save_path, Cres20_folder))

def save_step_imgs(ch, sh, save_path):
    """
    Save steganography images to their corresponding folders.
    """
    print(f"\nSaving Images to {save_path} ... ")
    
    _make_timestep_folders(save_path)
    save_images(ch, os.path.join(save_path, Chat_folder))
    save_images(sh, os.path.join(save_path, Shat_folder))
        

def save_images(imgs, folder):
    """ 
    Alternate save function.
    
    Parameters
    ----------
    imgs : tensor
        a tensor of tensor images
    folder : str
        the overall directory fo where to save the images
    """
    # Check on image render coloring
    maps = 'gray' if imgs.shape[1] == 1 else None
    for i in range(len(imgs)):
        plt.clf();
        plt.imshow(imgs[i].permute(1, 2, 0), cmap=maps);
        plt.axis("off");
        plt.savefig(f"{folder}/{i}.jpg", bbox_inches='tight');

def get_metrics(original, modified):
    """
    Get average image metrics for batches of color images.
    """
    #
    # Convert normalized tensors to numpy for skimage functions
    #
    original_np = (original * 255).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy()
    modified_np = (modified * 255).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy()
    #
    # Initialize metric lists
    #
    mse_list, psnr_list, ssim_list= [], [], []
    #
    # Calculate metrics
    #
    for orig, mod in zip(original_np, modified_np):
        # mse, psnr
        mse = MSE(orig, mod)
        psnr = PSNR(orig, mod, data_range=255)
        mse_list.append(mse)
        psnr_list.append(psnr)

        # compute SSIM
        ssim, _ = SSIM(orig, mod, multichannel=True, full=True, channel_axis=2, data_range=255)
        ssim_list.append(ssim)
    #
    # Calculate the bounded pixel error rate
    #
    bper_list = calc_bounded_pixel_error_rate(original, modified, eps=0.06)
    #
    # Calc ncc and authentication success rate
    #
    ncc_values = torch.zeros(modified.shape[0])
    asr_values = torch.zeros(modified.shape[0])
    for i in range(modified.shape[0]):
        ncc_values[i] = calc_ncc(original[i], modified[i])
        if ncc_values[i] > 0.95:
            asr_values[i] = 1.0

    return np.mean(mse_list), np.mean(psnr_list), np.mean(ssim_list), np.mean(bper_list), torch.mean(ncc_values).item(), sum(asr_values).item()/len(asr_values)
    

def calc_bounded_pixel_error_rate(im1, im2, eps=0.06):
    """
    Calculate bounded pixel error rate (BPER) of two images. A per pixel
    calculation of |im1_{c,i,j} - im2_{c,i,j}| <= eps for Ac,Ai,Aj in im1 and im2.
    Im1 and Im2 should have the same shape.

    initial value eps = 15/255 = 0.06

    Args:
        im1 (torch.Tensor) : image 1
        im2 (torch.Tensor) : image 2
        eps (float) : the epsilon bound of the allowable difference between each pixel
                      of the two images. 
    Returns:
        Percentage of pixels differences that are < eps.
    """
    #
    # Calculate over a batch
    #
    if len(im1.shape) > 3:
        return _calc_bounded_pixel_error_rate(im1, im2, eps)
    #
    # Calc correct pixels within the bound
    #
    error = [1 if abs(p1 - p2) > eps else 0 for p1, p2 in zip(im1.flatten(), im2.flatten())]
    #
    # Calc error rate
    #
    return sum(error)/im1.numel()
    

def _calc_bounded_pixel_error_rate(im1_batch, im2_batch, eps):
    """
    Calculate bounded pixel error rate (BPER) over batch of images.
    """
    #
    # Calc over entire batch
    #
    bper = []
    for im1, im2 in zip(im1_batch, im2_batch):
        bper.append(calc_bounded_pixel_error_rate(im1, im2, eps))

    return bper
    

def norm_data(data):
    """Normalize data to have mean=0 and standard_deviation=1 for tensor images.

    :param data: PyTorch tensor of any shape
    :type data: torch.Tensor
    :return: normalized batch of images
    :rtype: torch.tensor(float)
    """    
    #
    # Get mean and std
    #
    mean_data = torch.mean(data)
    std_data = torch.std(data, unbiased=True)
    #
    # normlize data
    #
    normalized_data = (data - mean_data) / std_data
    return normalized_data

def calc_ncc(data0, data1):
    """Normalized correlation coefficient (NCC) between two tensor datasets.

    :param data0: PyTorch tensors of the same size
    :type data0: torch.Tensor
    :param data1: PyTorch tensors of the same size
    :type data1: torch.Tensor
    :return: NCC comparison between two images
    :rtype: float [-1, 1]
    """    
    #
    # ensure float
    #
    data0 = data0.float()
    data1 = data1.float()
    #
    # normalize data
    #
    norm_data0 = norm_data(data0)
    norm_data1 = norm_data(data1)
    #
    # calculate ncc
    #
    ncc_value = torch.sum(norm_data0 * norm_data1) / (data0.numel() - 1)
    return ncc_value
        