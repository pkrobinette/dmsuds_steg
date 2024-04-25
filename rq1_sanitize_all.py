"""
RQ1: Evaluate sanitization of DM-SUDS against SUDS, 
Gaussian Noise, DCT Gaussin Noise. Save Images.
"""

from utils.utils import load_vae_suds,\
    load_udh,\
    load_ddh,\
    load_test_all,\
    use_lsb,\
    use_ddh,\
    use_udh,\
    add_gauss,\
    add_dct_noise,\
    add_gaussian_noise

from utils.DSM import DiffusionSanitizationModel

from utils.StegoPy import encode_img, decode_img
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch.nn as nn
import numpy as np
import random
import argparse
import torch
from tqdm import tqdm
import os
import argparse
import pandas as pd
import json
from PIL import Image

from utils.utils_compare_sanitize import save_eval_imgs, get_metrics

np.random.seed(44)
random.seed(44)
torch.manual_seed(44)

N = 40 # number of images to save


def get_args():
    """
    Get arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_path", default="results", help="Path/name of stats table.")
    args = parser.parse_args()

    return args

def eval_lsb_sanitize(covers, secrets, suds_model, diff_model, table_name):
    """
    Evaluate suds and dm-suds with containers made with lsb.
    """
    #
    # Get lsb containers
    #
    containers, reveal_secret = use_lsb(covers, secrets) # returned [0, 1]

    # get metrics
    mse_c, psnr_c, ssim_c, bper_c, ncc_c, asr_c = get_metrics(covers, containers)
    mse_s, psnr_s, ssim_s, bper_s, ncc_s, asr_s = get_metrics(secrets, reveal_secret)
    # ----------------------------------------------------------------
    #
    # Sanitize Suds, reveal, get metrics
    # -:- suds trained on [0, 255]
    # -:- decode_img returns in range [0, 255]
    #
    # ----------------------------------------------------------------
    print("\nSanitize with SUDS ...............................................")
    chat_suds = suds_model.sanitize(containers*255) # suds image input [0, 255]
    reveal_chat_suds = decode_img(chat_suds, train_mode=True)/255 # lsb works in [0, 255]
    
    # get metrics
    mse_chat_suds, psnr_chat_suds, ssim_chat_suds, bper_chat_suds, ncc_chat_suds, asr_chat_suds = get_metrics(covers, chat_suds)
    mse_secret_suds, psnr_secret_suds, ssim_secret_suds, bper_secret_suds, ncc_secret_suds, asr_secret_suds = get_metrics(secrets, reveal_chat_suds)

    # save images
    cres_suds = torch.clip(chat_suds - containers, 0, 1)
    save_path = "results/rq1/suds/lsb_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_suds[:N], reveal_chat_suds[:N], cres_suds[:N], torch.clip(cres_suds[:N]*5, 0, 1), torch.clip(cres_suds[:N]*10, 0, 1), torch.clip(cres_suds[:N]*20, 0, 1), save_path)
    # ----------------------------------------------------------------
    # 
    # Sanitize diffusion model and reveal
    # -:- decode_img returns in range [0, 255]
    #
    # ----------------------------------------------------------------
    print("\nSanitize with DM-SUDS ...............................................")
    chat_diff = diff_model.sanitize(containers)
    reveal_chat_diff = decode_img(chat_diff, train_mode=True)/255

    # get metrics
    mse_chat_diff, psnr_chat_diff, ssim_chat_diff, bper_chat_diff, ncc_chat_diff, asr_chat_diff = get_metrics(covers, chat_diff)
    mse_secret_diff, psnr_secret_diff, ssim_secret_diff, bper_secret_diff, ncc_secret_diff, asr_secret_diff = get_metrics(secrets, reveal_chat_diff)

    # save images
    cres_diff = torch.clip(chat_diff - containers, 0, 1)
    save_path = "results/rq1/diffusion_model/lsb_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_diff[:N], reveal_chat_diff[:N], cres_diff[:N],  torch.clip(cres_diff[:N]*5, 0, 1), torch.clip(cres_diff[:N]*10, 0, 1), torch.clip(cres_diff[:N]*20, 0, 1), save_path)
    # ----------------------------------------------------------------
    #
    # Sanitize Noise, reveal, get metrics
    #
    # ----------------------------------------------------------------
    print("\nSanitize with Noise ...............................................")
    # chat_noise = add_gauss(containers, sigma=0.1)
    chat_noise = add_gaussian_noise(containers, std=0.08)
    reveal_chat_noise = decode_img(chat_noise, train_mode=True)/255

    # get metrics
    mse_chat_noise, psnr_chat_noise, ssim_chat_noise, bper_chat_noise, ncc_chat_noise, asr_chat_noise = get_metrics(covers, chat_noise)
    mse_secret_noise, psnr_secret_noise, ssim_secret_noise, bper_secret_noise, ncc_secret_noise, asr_secret_noise = get_metrics(secrets, reveal_chat_noise)

    # save images
    cres_noise = torch.clip(chat_noise - containers, 0, 1)
    save_path = "results/rq1/noise/lsb_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_noise[:N], reveal_chat_noise[:N], cres_noise[:N], torch.clip(cres_noise[:N]*5,0,1), torch.clip(cres_noise[:N]*10,0,1), torch.clip(cres_noise[:N]*20,0,1), save_path)
    
    # ----------------------------------------------------------------
    #
    # Sanitize DCT Noise, reveal, get metrics
    #
    # ----------------------------------------------------------------
    print("\nSanitize with DCT Noise ...............................................")
    chat_dct = torch.clip(add_dct_noise(containers), 0, 1)
    reveal_chat_dct = decode_img(chat_dct, train_mode=True)/255

    # get metrics
    mse_chat_dct, psnr_chat_dct, ssim_chat_dct, bper_chat_dct, ncc_chat_dct, asr_chat_dct = get_metrics(covers, chat_dct)
    mse_secret_dct, psnr_secret_dct, ssim_secret_dct, bper_secret_dct, ncc_secret_dct, asr_secret_dct = get_metrics(secrets, reveal_chat_dct)

    # save images
    cres_dct = torch.clip(chat_dct - containers, 0, 1)
    save_path = "results/rq1/dct_noise/lsb_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_dct[:N], reveal_chat_dct[:N], cres_dct[:N], torch.clip(cres_dct[:N]*5,0,1), torch.clip(cres_dct[:N]*10,0,1), torch.clip(cres_dct[:N]*20,0,1), save_path)
    # ----------------------------------------------------------------
    #
    # Making tables
    #
    # ----------------------------------------------------------------

    print("\n\n -------- LSB Image Preservation -------- ")
    print("No Sanitization | Noise | DCT Noise | SUDS | DM-SUDS")
    print(f"MSE: {mse_c}   | {mse_chat_noise}  | {mse_chat_dct}  | {mse_chat_suds} | {mse_chat_diff} ")
    print(f"PSNR: {psnr_c} | {psnr_chat_noise} | {psnr_chat_dct} | {psnr_chat_suds}| {psnr_chat_diff} ")
    print(f"SSIM: {ssim_c} | {ssim_chat_noise} | {ssim_chat_dct} |{ssim_chat_suds} | {ssim_chat_diff} ")
    print(f"BPER: {bper_c} | {bper_chat_noise} | {bper_chat_dct} |{bper_chat_suds} | {bper_chat_diff} ")
    print(f"NCC: {ncc_c}   | {ncc_chat_noise}  | {ncc_chat_dct}  |{ncc_chat_suds}  | {ncc_chat_diff} ")
    print(f"ASR: {asr_c}   | {asr_chat_noise}  | {asr_chat_dct}  |{asr_chat_suds}  | {asr_chat_diff} ")
    
    print(" -------- LSB Secret Elimination -------- ")
    print(f"MSE: {mse_s}   |  {mse_secret_noise} | {mse_secret_dct}  | {mse_secret_suds} | {mse_secret_diff} ")
    print(f"PSNR: {psnr_s} | {psnr_secret_noise} | {psnr_secret_dct} |{psnr_secret_suds} | {psnr_secret_diff} ")
    print(f"SSIM: {ssim_s} | {ssim_secret_noise} | {ssim_secret_dct} |{ssim_secret_suds} | {ssim_secret_diff} ")
    print(f"BPER: {bper_s} | {bper_secret_noise} | {bper_secret_dct} |{bper_secret_suds} | {bper_secret_diff} ")
    print(f"NCC: {ncc_s}   | {ncc_secret_noise}  | {ncc_secret_dct}  |{ncc_secret_suds}  | {ncc_secret_diff} ")
    print(f"ASR: {asr_s}   | {asr_secret_noise}  | {asr_secret_dct}  |{asr_secret_suds}  | {asr_secret_diff} ")


    metrics_to_latex(table_name, "LSB",
                     mse_c, mse_chat_noise, mse_chat_dct, mse_chat_suds, mse_chat_diff, 
                     psnr_c, psnr_chat_noise, psnr_chat_dct, psnr_chat_suds, psnr_chat_diff, 
                     ssim_c, ssim_chat_noise, ssim_chat_dct, ssim_chat_suds, ssim_chat_diff,
                     bper_c, bper_chat_noise, bper_chat_dct, bper_chat_suds, bper_chat_diff,
                     ncc_c, ncc_chat_noise, ncc_chat_dct, ncc_chat_suds, ncc_chat_diff,
                     mse_s, mse_secret_noise, mse_secret_dct, mse_secret_suds, mse_secret_diff, 
                     psnr_s, psnr_secret_noise, psnr_secret_dct, psnr_secret_suds, psnr_secret_diff, 
                     ssim_s, ssim_secret_noise, ssim_secret_dct, ssim_secret_suds, ssim_secret_diff,
                     bper_s, bper_secret_noise, bper_secret_dct, bper_secret_suds, bper_secret_diff,
                     ncc_s, ncc_secret_noise, ncc_secret_dct, ncc_secret_suds, ncc_secret_diff,
                    )


def eval_ddh_sanitize(covers, secrets, suds_model, diff_model, HnetD, RnetD, table_name):
    """
    Evaluate suds and dm-suds with containers made with ddh
    """
    #
    # Get ddh containers
    #
    containers, reveal_secret = use_ddh(covers, secrets, HnetD, RnetD)

    # get metrics
    mse_c, psnr_c, ssim_c, bper_c, ncc_c, asr_c = get_metrics(covers, containers)
    mse_s, psnr_s, ssim_s, bper_s, ncc_s, asr_s = get_metrics(secrets, reveal_secret)
    # ------------------------------------------------------------------
    #
    # Sanitize Suds, reveal, get metrics
    # -:- suds trained on [0, 255]
    #
    # ------------------------------------------------------------------
    print("\nSanitize with SUDS ...............................................")
    chat_suds = suds_model.sanitize(containers*255) # suds image input [0, 255]
    with torch.no_grad():
        reveal_chat_suds = RnetD(chat_suds)
    
    # get metrics
    mse_chat_suds, psnr_chat_suds, ssim_chat_suds, bper_chat_suds, ncc_chat_suds, asr_chat_suds = get_metrics(covers, chat_suds)
    mse_secret_suds, psnr_secret_suds, ssim_secret_suds, bper_secret_suds, ncc_secret_suds, asr_secret_suds = get_metrics(secrets, reveal_chat_suds)

    # save images
    cres_suds = torch.clip(chat_suds - containers, 0, 1)
    save_path = "results/rq1/suds/ddh_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_suds[:N], reveal_chat_suds[:N], cres_suds[:N], torch.clip(cres_suds[:N]*5, 0, 1), torch.clip(cres_suds[:N]*10, 0, 1), torch.clip(cres_suds[:N]*20, 0, 1), save_path)
    # ------------------------------------------------------------------
    #
    # Sanitize diffusion model and reveal
    #
    # ------------------------------------------------------------------
    print("\nSanitize with DM-SUDS ...............................................")
    chat_diff = diff_model.sanitize(containers)
    with torch.no_grad():
        reveal_chat_diff = RnetD(chat_diff)

    # get metrics
    mse_chat_diff, psnr_chat_diff, ssim_chat_diff, bper_chat_diff, ncc_chat_diff, asr_chat_diff = get_metrics(covers, chat_diff)
    mse_secret_diff, psnr_secret_diff, ssim_secret_diff, bper_secret_diff, ncc_secret_diff, asr_secret_diff = get_metrics(secrets, reveal_chat_diff)

    # save images
    cres_diff = torch.clip(chat_diff - containers, 0, 1)
    save_path = "results/rq1/diffusion_model/ddh_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_diff[:N], reveal_chat_diff[:N], cres_diff[:N], torch.clip(cres_diff[:N]*5, 0, 1), torch.clip(cres_diff[:N]*10, 0, 1), torch.clip(cres_diff[:N]*20, 0, 1), save_path)
    # ------------------------------------------------------------------
    #
    # Sanitize Noise, reveal, get metrics
    #
    # ------------------------------------------------------------------
    print("\nSanitize with Noise ...............................................")
    chat_noise = add_gaussian_noise(containers, std=0.08)
    with torch.no_grad():
        reveal_chat_noise = RnetD(chat_noise)

    # get metrics
    mse_chat_noise, psnr_chat_noise, ssim_chat_noise, bper_chat_noise, ncc_chat_noise, asr_chat_noise = get_metrics(covers, chat_noise)
    mse_secret_noise, psnr_secret_noise, ssim_secret_noise, bper_secret_noise, ncc_secret_noise, asr_secret_noise = get_metrics(secrets, reveal_chat_noise)

    # save images
    cres_noise = torch.clip(chat_noise - containers, 0, 1)
    save_path = "results/rq1/noise/ddh_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_noise[:N], reveal_chat_noise[:N], cres_noise[:N], torch.clip(cres_noise[:N]*5, 0, 1), torch.clip(cres_noise[:N]*10, 0, 1), torch.clip(cres_noise[:N]*20, 0, 1), save_path)
    # ----------------------------------------------------------------
    #
    # Sanitize DCT Noise, reveal, get metrics
    #
    # ----------------------------------------------------------------
    print("\nSanitize with DCT Noise ...............................................")
    chat_dct = torch.clip(add_dct_noise(containers), 0, 1)
    with torch.no_grad():
        reveal_chat_dct = RnetD(chat_dct)

    # get metrics
    mse_chat_dct, psnr_chat_dct, ssim_chat_dct, bper_chat_dct, ncc_chat_dct, asr_chat_dct = get_metrics(covers, chat_dct)
    mse_secret_dct, psnr_secret_dct, ssim_secret_dct, bper_secret_dct, ncc_secret_dct, asr_secret_dct = get_metrics(secrets, reveal_chat_dct)

    # save images
    cres_dct = torch.clip(chat_dct - containers, 0, 1)
    save_path = "results/rq1/dct_noise/ddh_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_dct[:N], reveal_chat_dct[:N], cres_dct[:N], torch.clip(cres_dct[:N]*5,0,1), torch.clip(cres_dct[:N]*10,0,1), torch.clip(cres_dct[:N]*20,0,1), save_path)
    
    # ------------------------------------------------------------------
    #
    # Make tables
    #
    # ------------------------------------------------------------------
    print("\n\n -------- DDH Image Preservation -------- ")
    print("No Sanitization | Noise | DCT Noise | SUDS | DM-SUDS")
    print(f"MSE: {mse_c}   | {mse_chat_noise}  | {mse_chat_dct}  | {mse_chat_suds} | {mse_chat_diff} ")
    print(f"PSNR: {psnr_c} | {psnr_chat_noise} | {psnr_chat_dct} | {psnr_chat_suds}| {psnr_chat_diff} ")
    print(f"SSIM: {ssim_c} | {ssim_chat_noise} | {ssim_chat_dct} |{ssim_chat_suds} | {ssim_chat_diff} ")
    print(f"BPER: {bper_c} | {bper_chat_noise} | {bper_chat_dct} |{bper_chat_suds} | {bper_chat_diff} ")
    print(f"NCC: {ncc_c}   | {ncc_chat_noise}  | {ncc_chat_dct}  |{ncc_chat_suds}  | {ncc_chat_diff} ")
    print(f"ASR: {asr_c}   | {asr_chat_noise}  | {asr_chat_dct}  |{asr_chat_suds}  | {asr_chat_diff} ")
    
    print(" -------- DDH Secret Elimination -------- ")
    print(f"MSE: {mse_s}   |  {mse_secret_noise} | {mse_secret_dct}  | {mse_secret_suds} | {mse_secret_diff} ")
    print(f"PSNR: {psnr_s} | {psnr_secret_noise} | {psnr_secret_dct} |{psnr_secret_suds} | {psnr_secret_diff} ")
    print(f"SSIM: {ssim_s} | {ssim_secret_noise} | {ssim_secret_dct} |{ssim_secret_suds} | {ssim_secret_diff} ")
    print(f"BPER: {bper_s} | {bper_secret_noise} | {bper_secret_dct} |{bper_secret_suds} | {bper_secret_diff} ")
    print(f"NCC: {ncc_s}   | {ncc_secret_noise}  | {ncc_secret_dct}  |{ncc_secret_suds}  | {ncc_secret_diff} ")
    print(f"ASR: {asr_s}   | {asr_secret_noise}  | {asr_secret_dct}  |{asr_secret_suds}  | {asr_secret_diff} ")


    metrics_to_latex(table_name, "DDH",
                     mse_c, mse_chat_noise, mse_chat_dct, mse_chat_suds, mse_chat_diff, 
                     psnr_c, psnr_chat_noise, psnr_chat_dct, psnr_chat_suds, psnr_chat_diff, 
                     ssim_c, ssim_chat_noise, ssim_chat_dct, ssim_chat_suds, ssim_chat_diff,
                     bper_c, bper_chat_noise, bper_chat_dct, bper_chat_suds, bper_chat_diff,
                     ncc_c, ncc_chat_noise, ncc_chat_dct, ncc_chat_suds, ncc_chat_diff,
                     mse_s, mse_secret_noise, mse_secret_dct, mse_secret_suds, mse_secret_diff, 
                     psnr_s, psnr_secret_noise, psnr_secret_dct, psnr_secret_suds, psnr_secret_diff, 
                     ssim_s, ssim_secret_noise, ssim_secret_dct, ssim_secret_suds, ssim_secret_diff,
                     bper_s, bper_secret_noise, bper_secret_dct, bper_secret_suds, bper_secret_diff,
                     ncc_s, ncc_secret_noise, ncc_secret_dct, ncc_secret_suds, ncc_secret_diff,
                    )
    

def eval_udh_sanitize(covers, secrets, suds_model, diff_model, Hnet, Rnet, table_name):
    """
    Evaluate suds and dm-suds with containers made with udh
    """
    #
    # Get udh containers
    #
    containers, reveal_secret = use_udh(covers, secrets, Hnet, Rnet)

    # get metrics
    mse_c, psnr_c, ssim_c, bper_c, ncc_c, asr_c = get_metrics(covers, containers)
    mse_s, psnr_s, ssim_s, bper_s, ncc_s, asr_s = get_metrics(secrets, reveal_secret)
    # ---------------------------------------------------------------
    #
    # Sanitize Suds, reveal, get metrics
    # -:- suds trained on [0, 255]
    #
    # ---------------------------------------------------------------
    print("\nSanitize with SUDS ...............................................")
    chat_suds = suds_model.sanitize(containers*255) # suds image input [0, 255]
    with torch.no_grad():
        reveal_chat_suds = Rnet(chat_suds)
    
    # get metrics
    mse_chat_suds, psnr_chat_suds, ssim_chat_suds, bper_chat_suds, ncc_chat_suds, asr_chat_suds = get_metrics(covers, chat_suds)
    mse_secret_suds, psnr_secret_suds, ssim_secret_suds, bper_secret_suds, ncc_secret_suds, asr_secret_suds = get_metrics(secrets, reveal_chat_suds)

    # save images
    cres_suds = torch.clip(chat_suds - containers, 0, 1)
    save_path = "results/rq1/suds/udh_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_suds[:N], reveal_chat_suds[:N], cres_suds[:N], torch.clip(cres_suds[:N]*5, 0, 1), torch.clip(cres_suds[:N]*10, 0, 1), torch.clip(cres_suds[:N]*20, 0, 1), save_path)
    # ---------------------------------------------------------------
    #
    # Sanitize diffusion model and reveal
    #
    # ---------------------------------------------------------------
    print("\nSanitize with DM-SUDS ...............................................")
    chat_diff = diff_model.sanitize(containers)
    with torch.no_grad():
        reveal_chat_diff = Rnet(chat_diff)

    # get metrics
    mse_chat_diff, psnr_chat_diff, ssim_chat_diff, bper_chat_diff, ncc_chat_diff, asr_chat_diff = get_metrics(covers, chat_diff)
    mse_secret_diff, psnr_secret_diff, ssim_secret_diff, bper_secret_diff, ncc_secret_diff, asr_secret_diff = get_metrics(secrets, reveal_chat_diff)

    # save images
    cres_diff = torch.clip(chat_diff - containers, 0, 1)
    save_path = "results/rq1/diffusion_model/udh_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_diff[:N], reveal_chat_diff[:N], cres_diff[:N], torch.clip(cres_diff[:N]*5, 0, 1), torch.clip(cres_diff[:N]*10, 0, 1), torch.clip(cres_diff[:N]*20, 0, 1), save_path)
    # ---------------------------------------------------------------
    #
    # Sanitize Noise, reveal, get metrics
    #
    # ---------------------------------------------------------------
    print("\nSanitize with Noise ...............................................")
    chat_noise = add_gaussian_noise(containers, std=0.08)
    with torch.no_grad():
        reveal_chat_noise = Rnet(chat_noise)

    # get metrics
    mse_chat_noise, psnr_chat_noise, ssim_chat_noise, bper_chat_noise, ncc_chat_noise, asr_chat_noise = get_metrics(covers, chat_noise)
    mse_secret_noise, psnr_secret_noise, ssim_secret_noise, bper_secret_noise, ncc_secret_noise, asr_secret_noise = get_metrics(secrets, reveal_chat_noise)

    # save images
    cres_noise = torch.clip(chat_noise - containers, 0, 1)
    save_path = "results/rq1/noise/udh_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_noise[:N], reveal_chat_noise[:N], cres_noise[:N], torch.clip(cres_noise[:N]*5, 0, 1), torch.clip(cres_noise[:N]*10, 0, 1), torch.clip(cres_noise[:N]*20, 0, 1), save_path)
    # ----------------------------------------------------------------
    #
    # Sanitize DCT Noise, reveal, get metrics
    #
    # ----------------------------------------------------------------
    print("\nSanitize with DCT Noise ...............................................")
    chat_dct = torch.clip(add_dct_noise(containers), 0, 1)
    with torch.no_grad():
        reveal_chat_dct = Rnet(chat_dct)

    # get metrics
    mse_chat_dct, psnr_chat_dct, ssim_chat_dct, bper_chat_dct, ncc_chat_dct, asr_chat_dct = get_metrics(covers, chat_dct)
    mse_secret_dct, psnr_secret_dct, ssim_secret_dct, bper_secret_dct, ncc_secret_dct, asr_secret_dct = get_metrics(secrets, reveal_chat_dct)

    # save images
    cres_dct = torch.clip(chat_dct - containers, 0, 1)
    save_path = "results/rq1/dct_noise/udh_demo"
    save_eval_imgs(covers[:N], secrets[:N], containers[:N], reveal_secret[:N], chat_dct[:N], reveal_chat_dct[:N], cres_dct[:N], torch.clip(cres_dct[:N]*5,0,1), torch.clip(cres_dct[:N]*10,0,1), torch.clip(cres_dct[:N]*20,0,1), save_path)
    
    # ------------------------------------------------------------------
    #
    # Make tables
    #
    # ------------------------------------------------------------------
    print("\n\n -------- UDH Image Preservation -------- ")
    print("No Sanitization | Noise | DCT Noise | SUDS | DM-SUDS")
    print(f"MSE: {mse_c}   | {mse_chat_noise}  | {mse_chat_dct}  | {mse_chat_suds} | {mse_chat_diff} ")
    print(f"PSNR: {psnr_c} | {psnr_chat_noise} | {psnr_chat_dct} | {psnr_chat_suds}| {psnr_chat_diff} ")
    print(f"SSIM: {ssim_c} | {ssim_chat_noise} | {ssim_chat_dct} |{ssim_chat_suds} | {ssim_chat_diff} ")
    print(f"BPER: {bper_c} | {bper_chat_noise} | {bper_chat_dct} |{bper_chat_suds} | {bper_chat_diff} ")
    print(f"NCC: {ncc_c}   | {ncc_chat_noise}  | {ncc_chat_dct}  |{ncc_chat_suds}  | {ncc_chat_diff} ")
    print(f"ASR: {asr_c}   | {asr_chat_noise}  | {asr_chat_dct}  |{asr_chat_suds}  | {asr_chat_diff} ")
    
    print(" -------- UDH Secret Elimination -------- ")
    print(f"MSE: {mse_s}   |  {mse_secret_noise} | {mse_secret_dct}  | {mse_secret_suds} | {mse_secret_diff} ")
    print(f"PSNR: {psnr_s} | {psnr_secret_noise} | {psnr_secret_dct} |{psnr_secret_suds} | {psnr_secret_diff} ")
    print(f"SSIM: {ssim_s} | {ssim_secret_noise} | {ssim_secret_dct} |{ssim_secret_suds} | {ssim_secret_diff} ")
    print(f"BPER: {bper_s} | {bper_secret_noise} | {bper_secret_dct} |{bper_secret_suds} | {bper_secret_diff} ")
    print(f"NCC: {ncc_s}   | {ncc_secret_noise}  | {ncc_secret_dct}  |{ncc_secret_suds}  | {ncc_secret_diff} ")
    print(f"ASR: {asr_s}   | {asr_secret_noise}  | {asr_secret_dct}  |{asr_secret_suds}  | {asr_secret_diff} ")


    metrics_to_latex(table_name, "UDH",
                     mse_c, mse_chat_noise, mse_chat_dct, mse_chat_suds, mse_chat_diff, 
                     psnr_c, psnr_chat_noise, psnr_chat_dct, psnr_chat_suds, psnr_chat_diff, 
                     ssim_c, ssim_chat_noise, ssim_chat_dct, ssim_chat_suds, ssim_chat_diff,
                     bper_c, bper_chat_noise, bper_chat_dct, bper_chat_suds, bper_chat_diff,
                     ncc_c, ncc_chat_noise, ncc_chat_dct, ncc_chat_suds, ncc_chat_diff,
                     mse_s, mse_secret_noise, mse_secret_dct, mse_secret_suds, mse_secret_diff, 
                     psnr_s, psnr_secret_noise, psnr_secret_dct, psnr_secret_suds, psnr_secret_diff, 
                     ssim_s, ssim_secret_noise, ssim_secret_dct, ssim_secret_suds, ssim_secret_diff,
                     bper_s, bper_secret_noise, bper_secret_dct, bper_secret_suds, bper_secret_diff,
                     ncc_s, ncc_secret_noise, ncc_secret_dct, ncc_secret_suds, ncc_secret_diff,
                    )

    
def metrics_to_latex(table_name, steg_type,
                     mse_c, mse_chat_noise, mse_chat_dct, mse_chat_suds, mse_chat_diff, 
                     psnr_c, psnr_chat_noise, psnr_chat_dct, psnr_chat_suds, psnr_chat_diff, 
                     ssim_c, ssim_chat_noise, ssim_chat_dct, ssim_chat_suds, ssim_chat_diff,
                     bper_c, bper_chat_noise, bper_chat_dct, bper_chat_suds, bper_chat_diff,
                     ncc_c, ncc_chat_noise, ncc_chat_dct, ncc_chat_suds, ncc_chat_diff,
                     mse_s, mse_secret_noise, mse_secret_dct, mse_secret_suds, mse_secret_diff, 
                     psnr_s, psnr_secret_noise, psnr_secret_dct, psnr_secret_suds, psnr_secret_diff, 
                     ssim_s, ssim_secret_noise, ssim_secret_dct, ssim_secret_suds, ssim_secret_diff,
                     bper_s, bper_secret_noise, bper_secret_dct, bper_secret_suds, bper_secret_diff,
                     ncc_s, ncc_secret_noise, ncc_secret_dct, ncc_secret_suds, ncc_secret_diff):
    """
    Add to a latex table for results
    """
    #
    # Create a LaTeX table for chat metrics
    #
    if steg_type == "LSB":
        heading = f"""\\multirow{{2}}{{*}}{{$\\Hide$}} & \\multirow{{2}}{{*}}{{\\textbf{{Method}}}} & MSE & PSNR & SSIM & BPER & Sanitization & NCC \\\\
                      &  & (IP / SE) & (IP / SE) & (IP / SE) & (IP / SE) & (BPER IP/SE < 0.40) & (IP / SE)  \\\\ """
    else:
        heading = "\\midrule \n"
        
    latex_output = heading + f"""{steg_type} & None & {mse_c:.2f} / {mse_s:.2f} & {psnr_c:.2f} / {psnr_s:.2f} & {ssim_c:.2f} / {ssim_s:.2f} & {bper_c:.2f} / {bper_s:.2f} & {bper_c/bper_s:.2f} & {ncc_c:.2f} / {ncc_s:.2f} \\\\
            & Gaussian Noise & {mse_chat_noise:.2f} / {mse_secret_noise:.2f} & {psnr_chat_noise:.2f} / {psnr_secret_noise:.2f} & {ssim_chat_noise:.2f} / {ssim_secret_noise:.2f} & {bper_chat_noise:.2f} / {bper_secret_noise:.2f} & {bper_chat_noise/bper_secret_noise:.2f} & {ncc_chat_noise:.2f} / {ncc_secret_noise:.2f} \\\\
            & DCT Noise & {mse_chat_dct:.2f} / {mse_secret_dct:.2f} & {psnr_chat_dct:.2f} / {psnr_secret_dct:.2f} & {ssim_chat_dct:.2f} / {ssim_secret_dct:.2f} & {bper_chat_dct:.2f} / {bper_secret_dct:.2f} & {bper_chat_dct/bper_secret_dct:.2f} & {ncc_chat_dct:.2f} / {ncc_secret_dct:.2f} \\\\
            & SUDS & {mse_chat_suds:.2f} / {mse_secret_suds:.2f} & {psnr_chat_suds:.2f} / {psnr_secret_suds:.2f} & {ssim_chat_suds:.2f} / {ssim_secret_suds:.2f} & {bper_chat_suds:.2f} / {bper_secret_suds:.2f} & {bper_chat_suds/bper_secret_suds:.2f} & {ncc_chat_suds:.2f} / {ncc_secret_suds:.2f} \\\\
            & DM-SUDS & {mse_chat_diff:.2f} / {mse_secret_diff:.2f} & {psnr_chat_diff:.2f} / {psnr_secret_diff:.2f} & {ssim_chat_diff:.2f} / {ssim_secret_diff:.2f} & {bper_chat_diff:.2f} / {bper_secret_diff:.2f} & {bper_chat_diff/bper_secret_diff:.2f} & {ncc_chat_diff:.2f} / {ncc_secret_diff:.2f} \\\\
"""
    #
    # Combine and save to file
    #
    with open(table_name, 'a') as file:
        file.write(latex_output)


def main():
    """ main """
    args = get_args()
    #
    # Load data
    #
    test_images = load_test_all("cifar", pickle_path="test_set_04-2024")
    num = 10
    covers = test_images[:num]
    secrets = test_images[num:num*2]
    #
    # load models
    #
    diff_model = DiffusionSanitizationModel("models/diffusion_models/cifar10_uncond_50M_500K.pt")
    suds_model = load_vae_suds(channels=3, dataset="cifar")
    HnetD, RnetD = load_ddh(config="rgb_ddh")
    Hnet, Rnet = load_udh(config="rgb_udh")
    #
    # Make save table and evaluate
    #
    table = os.path.join(args.table_path, "rq1_table1.txt")
    with open(table, "w") as f:
        f.write("SUDS vs. Diffusion vs. Noise Model Test Stats\n")
        f.write("=====================================\n")
    eval_lsb_sanitize(covers, secrets, suds_model, diff_model, table)
    eval_ddh_sanitize(covers, secrets, suds_model, diff_model, HnetD, RnetD, table)
    eval_udh_sanitize(covers, secrets, suds_model, diff_model, Hnet, Rnet, table)

if __name__ == "__main__":
    main()
    os.system('say "your program has finished"')

    


