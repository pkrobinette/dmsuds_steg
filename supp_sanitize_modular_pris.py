"""
Evaluate DM-SUDS on imagenet with PRIS with a new model!
"""

from utils.utils import load_pris_model, use_pris
from utils.DSM_imagenet import DiffusionSanitizationModel
from utils.utils_compare_sanitize import save_eval_imgs, get_metrics
from utils.pris_utils import dwt, iwt
import argparse
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt
from data_imagenet import ImageNetDataset
import tqdm
import numpy as np
import sys

torch.manual_seed(4)


def get_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="datasets/ImageNet")
    parser.add_argument("--save_dir", type=str, default="results/supplement/modular_imagenet_pris")

    args = parser.parse_args()

    return args


def main():
    """ main """
    args = get_args()
    assert os.path.exists(args.root_dir) != 0, "Root data path does not exist."
    #
    # Load diffusion model
    #
    diff_model = DiffusionSanitizationModel("models/diffusion_models/imagenet64_uncond_vlb_100M_1500K.pt")
    #
    # Load DDH
    #
    net = load_pris_model()
    #
    # Load data
    # 
    print("Loading data ...")
    dataset = ImageNetDataset(args.root_dir, "test", num_images=500)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=2, shuffle=False, pin_memory=True)

    mse_chat_diff_all = 0
    psnr_chat_diff_all = 0
    ssim_chat_diff_all = 0
    bper_chat_diff_all = 0
    ncc_chat_diff_all = 0
    asr_chat_diff_all = 0
    
    mse_secret_diff_all = 0
    psnr_secret_diff_all = 0
    ssim_secret_diff_all = 0
    bper_secret_diff_all = 0
    ncc_secret_diff_all = 0
    asr_secret_diff_all = 0

    for i, covers in enumerate(tqdm.tqdm(data_loader)):
        perm = torch.randperm(covers.size(0))
        secrets = covers[perm]
        #
        # Make containers
        #
        containers, reveal_secrets, _ = use_pris(covers, secrets, net)
        #
        # Sanitize
        #
        print("\n\nSanitizing ...")
        out = diff_model.sanitize(containers)
        noise = torch.rand(containers.shape[0], 12, containers.shape[-1]//2, containers.shape[-1]//2)
        with torch.no_grad():
            out_en = net.pre_enhance(out)
            input_sanitized = dwt(out_en)
            sanitized_rev = torch.cat((input_sanitized, noise), 1)
            sanitized_image = net(sanitized_rev, rev=True)
            reveal_secret_sani = sanitized_image.narrow(1, 4 * 3,
                                         sanitized_image.shape[1] - 4 * 3)
            reveal_secret_sani = iwt(reveal_secret_sani)
            reveal_secret_sani = net.post_enhance(reveal_secret_sani)
        print("Saving images ...")
        cres = torch.clip(out - containers, 0, 1)
        if i == 0:
            save_eval_imgs(covers, secrets, containers, reveal_secrets, out, reveal_secret_sani, cres, torch.clip(cres*5, 0, 1), 
                           torch.clip(cres*10, 0, 1), torch.clip(cres*20, 0, 1), args.save_dir)
        # 
        # Get Metrics
        #
        containers = containers.clip(0, 1)
        out = out.clip(0, 1)
        reveal_secret_sani = reveal_secret_sani.clip(0, 1)
        
        print("Get metrics ...")
        mse_chat_diff, psnr_chat_diff, ssim_chat_diff, bper_chat_diff, ncc_chat_diff, asr_chat_diff = get_metrics(containers, out)
        mse_secret_diff, psnr_secret_diff, ssim_secret_diff, bper_secret_diff, ncc_secret_diff, asr_secret_diff = get_metrics(secrets, reveal_secret_sani)
        
        mse_chat_diff_all += mse_chat_diff
        psnr_chat_diff_all += psnr_chat_diff
        ssim_chat_diff_all += ssim_chat_diff
        bper_chat_diff_all += bper_chat_diff
        ncc_chat_diff_all += ncc_chat_diff
        asr_chat_diff_all += asr_chat_diff
        
        mse_secret_diff_all += mse_secret_diff
        psnr_secret_diff_all += psnr_secret_diff
        ssim_secret_diff_all += ssim_secret_diff
        bper_secret_diff_all += bper_secret_diff
        ncc_secret_diff_all += ncc_secret_diff
        asr_secret_diff_all += asr_secret_diff
        
    
    print("\n\n -------- PRIS Image Preservation -------- ")
    print(f"MSE:  {mse_chat_diff_all/len(data_loader)} ")
    print(f"PSNR: {psnr_chat_diff_all/len(data_loader)} ")
    print(f"SSIM: {ssim_chat_diff_all/len(data_loader)} ")
    print(f"BPER: {bper_chat_diff_all/len(data_loader)} ")
    print(f"NCC: {ncc_chat_diff_all/len(data_loader)} ")
    print(f"ASR: {asr_chat_diff_all/len(data_loader)} ")
    
    
    print(" -------- PRIS Secret Elimination -------- ")
    print(f"MSE: {mse_secret_diff_all/len(data_loader)} ")
    print(f"PSNR: {psnr_secret_diff_all/len(data_loader)} ")
    print(f"SSIM: {ssim_secret_diff_all/len(data_loader)} ")
    print(f"BPER: {bper_secret_diff_all/len(data_loader)} ")
    print(f"NCC: {ncc_secret_diff_all/len(data_loader)} ")
    print(f"ASR: {asr_secret_diff_all/len(data_loader)} ")
    print("======================================================")
    #
    # Save table
    #
    mse_image_preservation = mse_chat_diff_all / len(data_loader)
    psnr_image_preservation = psnr_chat_diff_all / len(data_loader)
    ssim_image_preservation = ssim_chat_diff_all / len(data_loader)
    bper_image_preservation = bper_chat_diff_all / len(data_loader)
    ncc_image_preservation = ncc_chat_diff_all / len(data_loader)
    asr_image_preservation = asr_chat_diff_all / len(data_loader)
    
    mse_secret_elimination = mse_secret_diff_all / len(data_loader)
    psnr_secret_elimination = psnr_secret_diff_all / len(data_loader)
    ssim_secret_elimination = ssim_secret_diff_all / len(data_loader)
    bper_secret_elimination = bper_secret_diff_all / len(data_loader)
    ncc_secret_elimination = ncc_secret_diff_all / len(data_loader)
    asr_secret_elimination = asr_secret_diff_all / len(data_loader)
    
    table = ""
    table += " -------- PRIS Image Preservation -------- \n"
    table += f"MSE:  {mse_image_preservation:.6f}\n"
    table += f"PSNR: {psnr_image_preservation:.6f}\n"
    table += f"SSIM: {ssim_image_preservation:.6f}\n"
    table += f"BPER: {bper_image_preservation:.6f}\n"
    table += f"NCC: {ncc_image_preservation:.6f}\n"
    table += f"ASR: {asr_image_preservation:.6f}\n\n"
    
    table += " -------- PRIS Secret Elimination -------- \n"
    table += f"MSE:  {mse_secret_elimination:.6f}\n"
    table += f"PSNR: {psnr_secret_elimination:.6f}\n"
    table += f"SSIM: {ssim_secret_elimination:.6f}\n"
    table += f"BPER: {bper_secret_elimination:.6f}\n"
    table += f"NCC: {ncc_secret_elimination:.6f}\n"
    table += f"ASR: {asr_secret_elimination:.6f}\n"
    table += "======================================================\n"
    
    with open(os.path.join(args.save_dir, 'modular_imagenet_pris_table.txt'), 'w') as file:
        file.write(table)


if __name__ == "__main__":
    main()
    

    
    