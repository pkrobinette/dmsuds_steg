from audiodiffusion import AudioDiffusion
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device)
import numpy as np
import utils as u
import matplotlib.pyplot as plt
import torchaudio
import utils_steg as u
from utils import get_embed_data, Loader
import pickle
import os
import sys
import pywt
import scipy.fftpack as fftpack
import pandas as pd



def sanitize(container, cover, audio_diffusion, Load):
    #
    # Create an image from the container
    # + normalize for diffusion model
    #
    image = Load.audio_to_image(container)
    image_norm = (image/255)*2-1  # model trained on [-1, 1]
    #
    # Generate random noise and add to the contianer image
    #
    t = 5 # timestep
    noise = torch.randn((1,1,256, 256), generator=generator, device="cpu")
    x_start = audio_diffusion.pipe.scheduler.add_noise(image_norm.unsqueeze(0), noise, torch.tensor([t]))
    #
    # diffusion model --> predict starting sample form noisy image
    #
    with torch.no_grad():
        model_output = audio_diffusion.pipe.unet(x_start, t)["sample"]
        pred = audio_diffusion.pipe.scheduler.step(
          model_output=model_output,
          timestep=t,
          sample=x_start
        )["pred_original_sample"]
    #
    # Unormlize predicted image and create an audio file again
    #
    image_pred = ((pred[0][0]+1)/2)*255 # reverse the normalization
    audio_pred = Load.image_to_audio(image_pred, cover)

    return audio_pred

def calc_metrics(container, sanitized, secret, pre_reveal_secret, post_reveal_secret):
    #
    # MSE between container and sanitized
    #
    dim = container.shape[0]
    if container.numel() != sanitized.numel():
        dim = min(container.shape[0], sanitized.shape[0])
    mse = torch.mean((container[:dim] - sanitized[:dim]) ** 2)

    def calculate_ber(seq1, seq2):
        #
        # get binary strings
        #
        binary_seq1 = ''.join(format(ord(i), '08b') for i in seq1)
        binary_seq2 = ''.join(format(ord(i), '08b') for i in seq2)
        #
        # Get differing bits
        #
        num_differing_bits = sum([1 if b1 != b2 else 0 for b1, b2 in zip(binary_seq1, binary_seq2)])
        #
        # calculate BER
        #
        ber = num_differing_bits / len(binary_seq1)
        
        return ber
        
    def _pad_text(s, reveal):
        #
        # If the lengths do not match, either chop or pad.
        #
        if len(reveal) != len(s):
            if len(reveal) > len(s):
                return reveal[:len(s)]
            else:
                return reveal + "0"*(len(s) - len(reveal))
        return reveal

    pre_reveal_secret = _pad_text(secret, pre_reveal_secret)
    post_reveal_secret = _pad_text(secret, post_reveal_secret)
    
      
    ber_pre = calculate_ber(secret, pre_reveal_secret)
    ber_post = calculate_ber(secret, post_reveal_secret)
    #
    # mse, bit error rate, bit error rate, removal rate
    #
    return mse, ber_pre, ber_post, 1-abs(2*ber_post-1)

def main():
    """
     main.
    """
    #
    # Get text secrets
    #
    URL1 = "http://shakespeare.mit.edu/twelfth_night/full.html"
    text_arr = []
    for url in [URL1]:
        arr = get_embed_data(url)
        text_arr.extend(arr)
    #
    # Create data loader
    #
    data_path = "data/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    Load = Loader(target_sample_rate = SAMPLE_RATE, num_samples = NUM_SAMPLES)
    #
    # Image file names
    #
    with open("data/UrbanSound8K/splits/test_sanitize_data_250.pkl", 'rb') as f:
        file_names = pickle.load(f)
    #
    # Load DM-SUDS
    #
    model_id = "teticio/audio-diffusion-256"
    audio_diffusion = AudioDiffusion(model_id=model_id)
    mel = audio_diffusion.pipe.mel
    #
    # Initialize metric values
    #
    lsb_mse = []
    lsb_ber_pre = []
    lsb_ber_post = []
    lsb_rr = []
    ss_mse = []
    ss_ber_pre = []
    ss_ber_post = []
    ss_rr = []
    dwt_mse = []
    dwt_ber_pre = []
    dwt_ber_post = []
    dwt_rr = []
    #
    # For each image and hiding method
    #
    for i, (im_path, secret) in enumerate(zip(file_names, text_arr[:len(file_names)])):
        print(f"Evaluating [{i+1}/{len(file_names)}]")
        #
        # Load audio
        #
        cover = Load.load_audio(os.path.join(data_path, im_path))
        #
        # LSB
        #
        container = u.lsb_embed_text_into_audio(cover, secret)[:len(cover)]
        sanitized = sanitize(container, cover, audio_diffusion, Load)
        pre_reveal_secret = u.lsb_extract_text_from_audio(container)
        post_reveal_secret = u.lsb_extract_text_from_audio(sanitized)
        mse, ber_pre, ber_post, rr = calc_metrics(container, sanitized, secret, pre_reveal_secret, post_reveal_secret)
        lsb_mse.append(mse)
        lsb_ber_pre.append(ber_pre)
        lsb_ber_post.append(ber_post)
        lsb_rr.append(rr)
        #
        # Spread Spectrum
        #
        container = u.ss_lsb_embed_text_into_audio(cover, secret)[:len(cover)]
        sanitized = sanitize(container, cover, audio_diffusion, Load)
        pre_reveal_secret = u.ss_lsb_extract_text_from_audio(container)
        post_reveal_secret = u.ss_lsb_extract_text_from_audio(sanitized)
        mse, ber_pre, ber_post, rr = calc_metrics(container, sanitized, secret, pre_reveal_secret, post_reveal_secret)
        ss_mse.append(mse)
        ss_ber_pre.append(ber_pre)
        ss_ber_post.append(ber_post)
        ss_rr.append(rr)
        #
        # DWT
        #
        container = u.dwt_embed_text_into_audio(cover, secret)[:len(cover)]
        sanitized = sanitize(container, cover, audio_diffusion, Load)
        pre_reveal_secret = u.dwt_extract_text_from_audio(container)
        post_reveal_secret = u.dwt_extract_text_from_audio(sanitized)
        mse, ber_pre, ber_post, rr = calc_metrics(container, sanitized, secret, pre_reveal_secret, post_reveal_secret)
        dwt_mse.append(mse)
        dwt_ber_pre.append(ber_pre)
        dwt_ber_post.append(ber_post)
        dwt_rr.append(rr)
    #
    # Print results
    #
    print("-------------------- Results ---------------------")
    print("\t LSB | DWT | SS")
    print(f"MSE: {round(np.mean(lsb_mse), 4)} | {round(np.mean(dwt_mse), 4)} | {round(np.mean(ss_mse), 4)}")
    print(f"Pre-BER: {round(np.mean(lsb_ber_pre), 4)} | {round(np.mean(dwt_ber_pre), 4)} | {round(np.mean(ss_ber_pre), 4)} ")
    print(f"Post-BER: {round(np.mean(lsb_ber_post), 4)} | {round(np.mean(dwt_ber_post), 4)} | {round(np.mean(ss_ber_post), 4)} ")
    print(f"RR: {round(np.mean(lsb_rr), 4)} | {round(np.mean(dwt_rr), 4)} | {round(np.mean(ss_rr), 4)} ")
    
    results = "-------------------- Results ---------------------\n"
    results += "\t LSB | DWT | SS \n"
    results += f"MSE: {round(np.mean(lsb_mse), 4)} | {round(np.mean(dwt_mse), 4)} | {round(np.mean(ss_mse), 4)}\n"
    results += f"Pre-BER: {round(np.mean(lsb_ber_pre), 4)} | {round(np.mean(dwt_ber_pre), 4)} | {round(np.mean(ss_ber_pre), 4)}\n"
    results += f"Post-BER: {round(np.mean(lsb_ber_post), 4)} | {round(np.mean(dwt_ber_post), 4)} | {round(np.mean(ss_ber_post), 4)}\n"
    results += f"RR: {round(np.mean(lsb_rr), 4)} | {round(np.mean(dwt_rr), 4)} | {round(np.mean(ss_rr), 4)}\n"
    
    with open("audio_sanitize_results.txt", "w") as text_file:
        text_file.write(results)
    #
    # Save data
    #
    df = pd.DataFrame({'lsb_mse': [np.mean(lsb_mse)],
                       'lsb_ber_pre': [np.mean(lsb_ber_pre)],
                       'lsb_ber_post': [np.mean(lsb_ber_post)],
                       'lsb_rr': [np.mean(lsb_rr)],
                       'ss_mse': [np.mean(ss_mse)],
                       'ss_ber_pre': [np.mean(ss_ber_pre)],
                       'ss_ber_post': [np.mean(ss_ber_post)],
                       'ss_rr': [np.mean(ss_rr)],
                       'dwt_mse': [np.mean(dwt_mse)],
                       'dwt_ber_pre': [np.mean(dwt_ber_pre)],
                       'dwt_ber_post': [np.mean(dwt_ber_post)],
                       'dwt_rr': [np.mean(dwt_rr)]},
                      index=[0])
    df_mean = df.round(4)
    df_mean.to_csv("audio_sanitize_results_means.csv")
