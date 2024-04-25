"""
RQ2: Evaluate DM-SUDS for different timesteps in the diffusion process. 
"""

from utils.utils import load_udh,\
    load_ddh,\
    load_data,\
    use_lsb,\
    use_ddh,\
    use_udh,\
    load_test_all

from utils.DSM import DiffusionSanitizationModel

from utils.StegoPy import encode_img, decode_img
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch.nn as nn
import numpy as np
import random
import torch
from tqdm import tqdm
import os
import argparse
import pandas as pd
import json
from PIL import Image

from utils.utils_compare_sanitize import save_eval_imgs, get_metrics, save_steg_imgs, save_step_imgs

np.random.seed(44)
random.seed(44)
torch.manual_seed(44)

N = 10
M = 20

def eval_lsb_sanitize(covers, secrets, diff_model):
    """
    Evaluate suds and dm-suds with containers made with lsb with various dm-suds timesteps.
    """
    print("Sanitizing LSB ..................................................\n\n")
    #
    # Get lsb containers
    #
    containers, reveal_secret = use_lsb(covers, secrets) # returned [0, 1]
    save_steg_imgs(covers[N:M], secrets[N:M], containers[N:M], reveal_secret[N:M], "results/rq2")
    # get metrics
    mse_c, psnr_c, ssim_c, bper_c, ncc_c, asr_c = get_metrics(covers, containers)
    mse_s, psnr_s, ssim_s, bper_s, ncc_s, asr_s = get_metrics(secrets, reveal_secret)
    #
    # Sanitize diffusion model and reveal
    # -:- decode_img returns in range [0, 255]
    #
    mse_chat = []
    mse_secret = []
    psnr_chat = []
    psnr_secret = []
    ssim_chat = []
    ssim_secret = []
    bper_chat = []
    bper_secret = []
    ncc_chat = []
    ncc_secret = []
    asr_chat = []
    asr_secret = []

    t_s = [x for x in range(25, 1001, 25)]
    # t_s.append(1000)    
    
    for t in t_s:
        chat_diff = diff_model.sanitize(containers, t=t)
        reveal_chat_diff = decode_img(chat_diff, train_mode=True)/255
        #
        # get metrics
        #
        mse_chat_diff, psnr_chat_diff, ssim_chat_diff, bper_chat_diff, ncc_chat_diff, asr_chat_diff = get_metrics(containers, chat_diff)
        mse_secret_diff, psnr_secret_diff, ssim_secret_diff, bper_secret_diff, ncc_secret_diff, asr_secret_diff = get_metrics(secrets, reveal_chat_diff)
        #
        # Record Metrics
        #
        mse_chat.append(mse_chat_diff)
        mse_secret.append(mse_secret_diff)
        psnr_chat.append(psnr_chat_diff)
        psnr_secret.append(psnr_secret_diff)
        ssim_chat.append(ssim_chat_diff)
        ssim_secret.append(ssim_secret_diff)
        bper_chat.append(bper_chat_diff)
        bper_secret.append(bper_secret_diff)
        ncc_chat.append(ncc_chat_diff)
        ncc_secret.append(ncc_secret_diff)
        asr_chat.append(asr_chat_diff)
        asr_secret.append(asr_secret_diff)
        # 
        # Save images
        # 
        if t < 525 or t > 975:
            save_path = f"results/rq2/lsb/step_{t}"
            save_step_imgs(chat_diff[N:M], reveal_chat_diff[N:M], save_path)
        else:
            print(f"timestep {t} ................\n")

    df = pd.DataFrame({'mse_chat': mse_chat, 'mse_secret': mse_secret,
                       'psnr_chat': psnr_chat, 'psnr_secret': psnr_secret,
                       'ssim_chat': ssim_chat, 'ssim_secret': ssim_secret,
                       'bper_chat': bper_chat, 'bper_secret': bper_secret,
                       'ncc_chat': ncc_chat, 'ncc_secret': ncc_secret,
                       'asr_chat': asr_chat, 'asr_secret': asr_secret
                      })
    df.to_csv('results/rq2/metrics_lsb.csv', index=False)
        

def eval_ddh_sanitize(covers, secrets, diff_model, HnetD, RnetD):
    """
    Evaluate suds and dm-suds with containers made with ddh with various dm-suds timesteps.
    """
    print("Sanitizing DDH ..................................................\n\n")
    #
    # Get ddh containers
    #
    containers, reveal_secret = use_ddh(covers, secrets, HnetD, RnetD)

    # get metrics
    mse_c, psnr_c, ssim_c, bper_c, ncc_c, asr_c = get_metrics(covers, containers)
    mse_s, psnr_s, ssim_s, bper_s, ncc_s, asr_s = get_metrics(secrets, reveal_secret)
    #
    # Sanitize diffusion model and reveal
    # -:- decode_img returns in range [0, 255]
    #
    mse_chat = []
    mse_secret = []
    psnr_chat = []
    psnr_secret = []
    ssim_chat = []
    ssim_secret = []
    bper_chat = []
    bper_secret = []
    ncc_chat = []
    ncc_secret = []
    asr_chat = []
    asr_secret = []

    t_s = [x for x in range(25, 1001, 25)]
    
    for t in t_s:
        chat_diff = diff_model.sanitize(containers, t=t)
        with torch.no_grad():
            reveal_chat_diff = RnetD(chat_diff)
        #
        # get metrics
        #
        mse_chat_diff, psnr_chat_diff, ssim_chat_diff, bper_chat_diff, ncc_chat_diff, asr_chat_diff = get_metrics(containers, chat_diff)
        mse_secret_diff, psnr_secret_diff, ssim_secret_diff, bper_secret_diff, ncc_secret_diff, asr_secret_diff = get_metrics(secrets, reveal_chat_diff)
        #
        # Record Metrics
        #
        mse_chat.append(mse_chat_diff)
        mse_secret.append(mse_secret_diff)
        psnr_chat.append(psnr_chat_diff)
        psnr_secret.append(psnr_secret_diff)
        ssim_chat.append(ssim_chat_diff)
        ssim_secret.append(ssim_secret_diff)
        bper_chat.append(bper_chat_diff)
        bper_secret.append(bper_secret_diff)
        ncc_chat.append(ncc_chat_diff)
        ncc_secret.append(ncc_secret_diff)
        asr_chat.append(asr_chat_diff)
        asr_secret.append(asr_secret_diff)
        # 
        # Save images
        # 
        if t < 525 or t > 975:
            save_path = f"results/rq2/ddh/step_{t}"
            save_step_imgs(chat_diff[N:M], reveal_chat_diff[N:M], save_path)
        else:
            print(f"timestep {t} ................\n")

    df = pd.DataFrame({'mse_chat': mse_chat, 'mse_secret': mse_secret,
                       'psnr_chat': psnr_chat, 'psnr_secret': psnr_secret,
                       'ssim_chat': ssim_chat, 'ssim_secret': ssim_secret,
                       'bper_chat': bper_chat, 'bper_secret': bper_secret,
                       'ncc_chat': ncc_chat, 'ncc_secret': ncc_secret,
                       'asr_chat': asr_chat, 'asr_secret': asr_secret
                      })
    df.to_csv('results/rq2/metrics_ddh.csv', index=False)
    

def eval_udh_sanitize(covers, secrets, diff_model, Hnet, Rnet):
    """
    Evaluate suds and dm-suds with containers made with udh with various dm-suds timesteps.
    """
    print("Sanitizing UDH ..................................................\n\n")
    #
    # Get udh containers
    #
    containers, reveal_secret = use_udh(covers, secrets, Hnet, Rnet)

    # get metrics
    mse_c, psnr_c, ssim_c, bper_c, ncc_c, asr_c = get_metrics(covers, containers)
    mse_s, psnr_s, ssim_s, bper_s, ncc_s, asr_s = get_metrics(secrets, reveal_secret)
    #
    # Sanitize diffusion model and reveal
    # -:- decode_img returns in range [0, 255]
    #
    mse_chat = []
    mse_secret = []
    psnr_chat = []
    psnr_secret = []
    ssim_chat = []
    ssim_secret = []
    bper_chat = []
    bper_secret = []
    ncc_chat = []
    ncc_secret = []
    asr_chat = []
    asr_secret = []

    t_s = [x for x in range(25, 1001, 25)]
    
    for t in t_s:
        chat_diff = diff_model.sanitize(containers, t=t)
        with torch.no_grad():
            reveal_chat_diff = Rnet(chat_diff)
        #
        # get metrics
        #
        mse_chat_diff, psnr_chat_diff, ssim_chat_diff, bper_chat_diff, ncc_chat_diff, asr_chat_diff = get_metrics(containers, chat_diff)
        mse_secret_diff, psnr_secret_diff, ssim_secret_diff, bper_secret_diff, ncc_secret_diff, asr_secret_diff = get_metrics(secrets, reveal_chat_diff)
        #
        # Record Metrics
        #
        mse_chat.append(mse_chat_diff)
        mse_secret.append(mse_secret_diff)
        psnr_chat.append(psnr_chat_diff)
        psnr_secret.append(psnr_secret_diff)
        ssim_chat.append(ssim_chat_diff)
        ssim_secret.append(ssim_secret_diff)
        bper_chat.append(bper_chat_diff)
        bper_secret.append(bper_secret_diff)
        ncc_chat.append(ncc_chat_diff)
        ncc_secret.append(ncc_secret_diff)
        asr_chat.append(asr_chat_diff)
        asr_secret.append(asr_secret_diff)
        # 
        # Save images
        # 
        if t < 525 or t > 975:
            save_path = f"results/rq2/udh/step_{t}"
            save_step_imgs(chat_diff[N:M], reveal_chat_diff[N:M], save_path)
        else:
            print(f"timestep {t} ................\n")

    df = pd.DataFrame({'mse_chat': mse_chat, 'mse_secret': mse_secret,
                       'psnr_chat': psnr_chat, 'psnr_secret': psnr_secret,
                       'ssim_chat': ssim_chat, 'ssim_secret': ssim_secret,
                       'bper_chat': bper_chat, 'bper_secret': bper_secret,
                       'ncc_chat': ncc_chat, 'ncc_secret': ncc_secret,
                       'asr_chat': asr_chat, 'asr_secret': asr_secret
                      })
    df.to_csv('results/rq2/metrics_udh.csv', index=False)


def main():
    #
    # Load data
    #
    test_images = load_test_all("cifar", pickle_path="test_set_04-2024")
    num = 500
    covers = test_images[:num]
    secrets = test_images[num:2*num]
    HnetD, RnetD = load_ddh(config="rgb_ddh")
    Hnet, Rnet = load_udh(config="rgb_udh")
    #
    # load models
    #
    diff_model = DiffusionSanitizationModel("models/diffusion_models/cifar10_uncond_50M_500K.pt")
    #
    # Evaluate on timesteps
    #
    eval_lsb_sanitize(covers, secrets, diff_model)
    eval_ddh_sanitize(covers, secrets, diff_model, HnetD, RnetD)
    eval_udh_sanitize(covers, secrets, diff_model, Hnet, Rnet)

if __name__ == "__main__":
    main()
    

    


