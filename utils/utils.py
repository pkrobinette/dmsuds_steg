"""
Utility functions used in the main folder.
"""
import numpy
import os
import glob
import time
import argparse
from PIL import Image
# from rawkit import raw
import torchvision
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import yaml
from yaml.loader import SafeLoader
from .HidingUNet import UnetGenerator
from .RevealNet import RevealNet
from .StegoPy import encode_msg, decode_msg, encode_img, decode_img
from .pris_model import PRIS
from .pris_utils import embed_attack, dwt, iwt
from .vae import CNN_VAE
from .classifier import Classifier
import torch.nn.init as init
import itertools
from tqdm import tqdm
from torch.nn.functional import normalize
from skimage.util import random_noise
import numpy as np
import random
import pickle
import scipy.fftpack as fftpack

np.random.seed(4)
random.seed(4)
torch.manual_seed(8)

TRANSFORMS_GRAY = transforms.Compose([ 
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize([32, 32]), 
                transforms.ToTensor(),
            ])

TRANSFORMS_RGB = transforms.Compose([
                transforms.Resize([32, 32]), 
                transforms.ToTensor(),
            ])  

SUDS_CONFIG_PATH = "configs/" # CHANGE IF DIFFERENT


# Custom weights initialization called on netG and netD
def weights_init(m):
    """
    Init weights.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.0) 
        m.bias.data.fill_(0)


def load_data(dataset, batch_size=128):
    """
    Load a dataset for training and testing.
    
    Parameters
    ----------
    dataset : str
        Indicate which dataset to load (mnist or cifar)
    batch_size : int
        The number of images in each batch
    
    Returns
    -------
    train_loader : DataLoader
        Training set
    test_loader : DataLoader
        Test set
    """
    assert (dataset in ["mnist", "cifar"]), "Invalid dataset key; mnist or cifar"
    
    if dataset == "mnist":
        trainset = datasets.MNIST(root='data', train=True,
                                            download=True, transform=TRANSFORMS_GRAY)
        train_loader = DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=False)
        testset = datasets.MNIST(root='data', train=False,
                                           download=True, transform=TRANSFORMS_GRAY)
        test_loader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
    else:
        trainset = datasets.CIFAR10(root='data', train=True,
                                            download=True, transform=TRANSFORMS_RGB)
        train_loader = DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=False)
        testset = datasets.CIFAR10(root='data', train=False,
                                           download=True, transform=TRANSFORMS_RGB)
        test_loader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
    
    return train_loader, test_loader

def load_data_all(dataset):
    """
    Load a dataset for training and testing.
    
    Parameters
    ----------
    dataset : str
        Indicate which dataset to load (mnist or cifar)
    batch_size : int
        The number of images in each batch
    
    Returns
    -------
    trainset : tensor
        Training set
    testset : tensor
        Test set
    """
    assert (dataset in ["mnist", "cifar"]), "Invalid dataset key; mnist or cifar"
    
    if dataset == "mnist":
        trainset = datasets.MNIST(root='data', train=True,
                                            download=True, transform=TRANSFORMS_GRAY)
        testset = datasets.MNIST(root='data', train=False,
                                           download=True, transform=TRANSFORMS_GRAY)
    else:
        trainset = datasets.CIFAR10(root='data', train=True,
                                            download=True, transform=TRANSFORMS_RGB)
        testset = datasets.CIFAR10(root='data', train=False,
                                           download=True, transform=TRANSFORMS_RGB)

    alltest = torch.empty([len(testset), 3, 32, 32])
    alltrain = torch.empty([len(trainset), 3, 32, 32])

    # Iterate over the test dataset and fill up the tensor
    for i, (img, _) in enumerate(testset):
        alltest[i] = img
        
    # Iterate over the train dataset and fill up the tensor
    for i, (img, _) in enumerate(trainset):
        alltrain[i] = img
    
    return alltrain, alltest


def load_test_all(dataset, pickle_path, num_images=2000, imsize=32):
    """
    Load a dataset for training and testing.
    
    Parameters
    ----------
    dataset : str
        Indicate which dataset to load (mnist or cifar)
    batch_size : int
        The number of images in each batch
    
    Returns
    -------
    trainset : tensor
        Training set
    testset : tensor
        Test set
    """
    assert (dataset in ["mnist", "cifar"]), "Invalid dataset key; mnist or cifar"
    
    if dataset == "mnist":
        testset = datasets.MNIST(root='data', train=False,
                                           download=True, transform=TRANSFORMS_GRAY)
    else:
        testset = datasets.CIFAR10(root='data', train=False,
                                           download=True, transform=TRANSFORMS_RGB)
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            indices = pickle.load(f)
    else:
        #
        # create new indices
        #
        print(f"Saving indices to {pickle_path} ... " )
        indices = torch.randperm(len(testset))[:num_images].tolist()
        with open(pickle_path, 'wb') as f:
            pickle.dump(indices, f)
            
    alltest = torch.empty([num_images, 3, imsize, imsize])

    # Iterate over the test dataset and fill up the tensor
    for i, idx in enumerate(indices):
        image, _ = testset[idx]
        alltest[i] = image

    return alltest


def load_pris_model(config="pris"):
    """
    Load the trained model for PRIS steganography.

    Parameters
    ----------
    config : str
        The name of the config file in the SUDS_CONFIG_PATH dir. 
        If a different dir, change global var above.

    Returns
    -------
    pris model
    """
    #
    # check config
    #
    if ".yml" not in config or ".yaml" not in config:
        config += ".yml"
    path = SUDS_CONFIG_PATH + config
    assert (os.path.exists(path)), "config path does not exist. Try again."
    # load yaml file
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    #
    # initialize model
    #
    net = PRIS(in_1=3, in_2=3)

    try:
        state_dicts = torch.load(data["model_path"])
    except:
        state_dicts = torch.load(data["model_path"], map_location=torch.device('cpu'))
        
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
            
    return net
    
    


def load_ddh(config="gray_ddh"):
    """
    Load the trained ddh networks for ddh steganography.
    
    Parameters
    ----------
    config : str
        The name of the config file in the SUDS_CONFIG_PATH dir. 
        If a different dir, change global var above.
    
    Returns
    -------
    ddh model
    """
    #
    # check config
    #
    if ".yml" not in config or ".yaml" not in config:
        config += ".yml"
    path = SUDS_CONFIG_PATH + config
    assert (os.path.exists(path)), "config path does not exist. Try again."
    # load yaml file
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    #
    # Initialize models
    #
    norm_layer = nn.BatchNorm2d
    HnetD = UnetGenerator(input_nc=data["channel_secret"]*data["num_secret"]+data["channel_cover"]*data["num_cover"], output_nc=data["channel_cover"]*data["num_cover"], num_downs=data["num_downs"], norm_layer=norm_layer, output_function=nn.Sigmoid)
    RnetD = RevealNet(input_nc=data["channel_cover"]*data["num_cover"], output_nc=data["channel_secret"]*data["num_secret"], nhf=64, norm_layer=norm_layer, output_function=nn.Sigmoid)
    HnetD.apply(weights_init)
    RnetD.apply(weights_init)
    #
    # Apply DDH saved info
    #
    ## WITH RESAVE
    try:
        H_state_diff = torch.load(data["test_diff"] + "/HnetD_checkpoint.tar")
        R_state_diff = torch.load(data["test_diff"] + "/RnetD_checkpoint.tar")
    except:
        checkpoint_diff = data["test_diff"] + "/checkPoints/" + "checkpoint.pth.tar"
        checkpoint_diff = torch.load(checkpoint_diff, map_location=torch.device('cpu'))
        H_state_diff = {}
        R_state_diff = {}
        for k in checkpoint_diff["H_state_dict"].keys():
            name = k.replace("module.", "")
            H_state_diff[name] = checkpoint_diff["H_state_dict"][k]
        
        for l in checkpoint_diff["R_state_dict"].keys():
            name = l.replace("module.", "")
            R_state_diff[name] = checkpoint_diff["R_state_dict"][l]
        
    HnetD.load_state_dict(H_state_diff)
    RnetD.load_state_dict(R_state_diff)
    
    # print("Re SAVE: ")
    # n = "ddh_mnist" if "gray" in config else "ddh_cifar"
    # torch.save(HnetD.state_dict(), f'models/steg/{n}/HnetD_checkpoint.tar')
    # torch.save(RnetD.state_dict(), f'models/steg/{n}/RnetD_checkpoint.tar')
    
    
    print(f"Finished loading DDH Models...")
    
    return HnetD, RnetD

# def load_ddh_imagenet(config="rgb_ddh_imagenet"):
#     #
#     # check config
#     #
#     if ".yml" not in config or ".yaml" not in config:
#         config += ".yml"
#     path = SUDS_CONFIG_PATH + config
#     assert (os.path.exists(path)), "config path does not exist. Try again."
#     # load yaml file
#     with open(path) as f:
#         data = yaml.load(f, Loader=SafeLoader)
#     #
#     # Initialize models
#     #
#     norm_layer = nn.BatchNorm2d
#     HnetD = UnetGenerator(input_nc=data["channel_secret"]*data["num_secret"]+data["channel_cover"]*data["num_cover"], output_nc=data["channel_cover"]*data["num_cover"], num_downs=data["num_downs"], norm_layer=norm_layer, output_function=nn.Sigmoid)
#     RnetD = RevealNet(input_nc=data["channel_cover"]*data["num_cover"], output_nc=data["channel_secret"]*data["num_secret"], nhf=64, norm_layer=norm_layer, output_function=nn.Sigmoid)
#     HnetD.apply(weights_init)
#     RnetD.apply(weights_init)
#     #
#     # Apply DDH saved info
#     #
#     ## WITH RESAVE
#     try:
#         H_state_diff = torch.load(data["test_diff"] + "/HnetD_checkpoint.tar")
#         R_state_diff = torch.load(data["test_diff"] + "/RnetD_checkpoint.tar")
#     except:
#         checkpoint_diff = data["test_diff"] + "/checkPoints/" + "checkpoint.pth.tar"
#         checkpoint_diff = torch.load(checkpoint_diff, map_location=torch.device('cpu'))
#         H_state_diff = {}
#         R_state_diff = {}
#         for k in checkpoint_diff["H_state_dict"].keys():
#             name = k.replace("module.", "")
#             H_state_diff[name] = checkpoint_diff["H_state_dict"][k]
        
#         for l in checkpoint_diff["R_state_dict"].keys():
#             name = l.replace("module.", "")
#             R_state_diff[name] = checkpoint_diff["R_state_dict"][l]
        
#     HnetD.load_state_dict(H_state_diff)
#     RnetD.load_state_dict(R_state_diff)
    
#     # print("Re SAVE: ")
#     # n = "ddh_mnist" if "gray" in config else "ddh_cifar"
#     # torch.save(HnetD.state_dict(), f'models/steg/{n}/HnetD_checkpoint.tar')
#     # torch.save(RnetD.state_dict(), f'models/steg/{n}/RnetD_checkpoint.tar')
    
    
#     print(f"Finished loading MNIST DDH Models...")
    
#     return HnetD, RnetD


#     if opt.checkpoint != "":
#         if opt.checkpoint_diff != "":
#             checkpoint = torch.load(opt.checkpoint)
#             Hnet.load_state_dict(checkpoint['H_state_dict'])
#             Rnet.load_state_dict(checkpoint['R_state_dict'])


def load_udh(config="gray_udh"):
    """
    Load the trained udh networks for udh steganography.
    
    Parameters
    ----------
    config : str
        The name of the config file in the SUDS_CONFIG_PATH dir. 
        If a different dir, change global var above.
    
    Returns
    -------
    udh model
    """
    #
    # Set up parameters
    #
    if ".yml" not in config or ".yaml" not in config:
        config += ".yml"
    path = SUDS_CONFIG_PATH + config
    assert (os.path.exists(path)), "config path does not exist. Try again."
    # load yaml file
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    #
    # Initialize models
    #
    norm_layer = nn.BatchNorm2d
    Hnet = UnetGenerator(input_nc=data["channel_secret"]*data["num_secret"], output_nc=data["channel_cover"]*data["num_cover"], num_downs=data["num_downs"], norm_layer=norm_layer, output_function=nn.Tanh)
    Rnet = RevealNet(input_nc=data["channel_cover"]*data["num_cover"], output_nc=data["channel_secret"]*data["num_secret"], nhf=64, norm_layer=norm_layer, output_function=nn.Sigmoid)
    Hnet.apply(weights_init)
    Rnet.apply(weights_init)
    
    #
    # Apply saved checkpoint and weights
    #
    ## WITH RESAVE
    try:
        H_state = torch.load(data["test"] + "/Hnet_checkpoint.tar")
        R_state = torch.load(data["test"] + "/Rnet_checkpoint.tar")
    except:
        checkpoint = data["test"] + "/checkPoints/" + "checkpoint.pth.tar"
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        H_state = {}
        R_state = {}
        for k in checkpoint["H_state_dict"].keys():
            name = k.replace("module.", "")
            H_state[name] = checkpoint["H_state_dict"][k]
        
        for l in checkpoint["R_state_dict"].keys():
            name = l.replace("module.", "")
            R_state[name] = checkpoint["R_state_dict"][l]
    
    Hnet.load_state_dict(H_state)
    Rnet.load_state_dict(R_state)
    
    # print("re SAVE: ")
    # n = "udh_mnist" if "gray" in config else "udh_cifar"
    # torch.save(Hnet.state_dict(), f'models/steg/{n}/Hnet_checkpoint.tar')
    # torch.save(Rnet.state_dict(), f'models/steg/{n}/Rnet_checkpoint.tar')
    
    Hnet.eval()
    Rnet.eval()
    
    print(f"Finished loading UDH Models...")
    
    return Hnet, Rnet

def load_vae_suds(channels=1, k_num=128, z_size=128, im_size=32, dataset="mnist"):
    """ 
    Load vae sanitization model, SUDS
    
    Parameters
    ----------
    channels : int
        Number of color channels, 3 = rgb, 1 = grayscale
    k_num : int
        kernal number used during training
    z_size : int
        latent space size used during training
    im_size : int
        image size (32x32)
        
    Returns
    -------
    vae_model : vae
    """
    # 
    # Get intended load directory
    # 
    name = "_"+str(z_size)+"/"
    # if z_size == 128:
    #     name = "/"
    path = "models/sanitization/suds_"+dataset+name+"model.pth"
    print(f"VAE load using --> {path}")
    assert (os.path.exists(path)), "Model does not exist. Try again."
    #
    # Load intended model
    #
    vae_model = CNN_VAE(c_in=channels, k_num=k_num, z_size=z_size, im_size=im_size);
    try:
        vae_model.load_state_dict(torch.load(path));
    except:
        vae_model.load_state_dict(torch.load(path, map_location='cpu'));
    vae_model.eval();
    
    return vae_model

def load_classifier(c_in=1, k_num=128, class_num=10, im_size=32, name="mnist_ddh_marked"):
    """
    Load a classifier trained with poisoned data.
    
    Parameters
    ----------
    name : str
        The name of the save directory within models/data_poison/.
    """
    # 
    # Get intended load directory
    # 
    path = "models/data_poison/"+name+"/model.pth"
    print(f"Data Poison load using --> {path}")
    assert (os.path.exists(path)), "Model does not exist. Try again."
    #
    # Load intended model
    #
    model = Classifier(c_in=c_in, k_num=k_num, class_num=class_num, im_size=im_size);
    try:
        model.load_state_dict(torch.load(path));
    except:
        model.load_state_dict(torch.load(path, map_location='cpu'));
    model.eval();
    
    return model

def use_lsb(covers, secrets):
    """
    Create containers using lsb hide method.
    
    Parameters
    ----------
    covers : tensor
        cover images
    secrets : tensor
        secrets to hide inside covers
    
    Returns
    -------
    containers : tensor [0, 1]
        secrets hidden in covers
    reveal_secrets : tensor [0, 1]
        recovered secrets from container
    """
    if covers.max() <= 1:
        covers = covers.clone().detach()*255
        secrets = secrets.clone().detach()*255
    #
    # Steg hide
    #
    containers = encode_img(covers, secrets, train_mode=True) # steg function is on pixels [0, 255]
    reveal_secret = decode_img(containers, train_mode=True)

    return containers/255, reveal_secret/255


def use_ddh(covers, secrets, HnetD, RnetD):
    """
    Create containers using ddh hide method.
    
    Parameters
    ----------
    covers : tensor
        cover images
    secrets : tensor
        secrets to hide inside covers
    HnetD : ddh hide
    RnetD : ddh reveal
    
    
    Returns
    -------
    containers : tensor
        secrets hidden in covers
    reveal_secret: tensor
        The secret revealed from the container
    """
    if covers.max() > 1:
        covers = covers.clone().detach()/255
    if secrets.max() > 1:
        secrets = secrets.clone().detach()/255
    #
    # Steg Hide
    #
    H_input = torch.cat((covers, secrets), dim=1)
    with torch.no_grad():
        containers = HnetD(H_input)
    
    with torch.no_grad():
        reveal_secret = RnetD(containers)

    cont_max = containers.max()
    cont_min = containers.min()
    secret_max = reveal_secret.max()
    secret_min = reveal_secret.min()
        
    return (containers - cont_min) / (cont_max - cont_min), (reveal_secret - secret_min) / (secret_max - secret_min)
    
    # return torch.clip(containers, 0, 1), torch.clip(reveal_secret, 0, 1)


def use_udh(covers, secrets, Hnet, Rnet):
    """
    Create containers using udh hide method.
    
    Parameters
    ----------
    covers : tensor
        cover images
    secrets : tensor
        secrets to hide inside covers
    Hnet : udh hide
    Rnet : udh reveal
    
    
    Returns
    -------
    containers : tensor
        secrets hidden in covers
    reveal_secret: tensor
        The secret revealed from the container
    """
    if covers.max() > 1:
        covers = covers.clone().detach()/255
    if secrets.max() > 1:
        secrets = secrets.clone().detach()/255
    try:
        _, c, h, w = covers.shape
    except:
        c, h, w = covers.shape
    #
    # Steg Hide
    #
    with torch.no_grad():
        C_res = Hnet(secrets)
        containers = C_res + covers
        reveal_secret = Rnet(containers)        
    
    return torch.clip(containers, 0, 1), torch.clip(reveal_secret, 0, 1)

def use_pris(covers, secrets, net):
    """
    Create containers using PRIS hide method.
    
    Parameters
    ----------
    covers : tensor
        cover images
    secrets : tensor
        secrets to hide inside covers
    Hnet : udh hide
    Rnet : udh reveal
    
    
    Returns
    -------
    containers : tensor
        secrets hidden in covers
    reveal_secret: tensor
        The secret revealed from the container
    """
    if covers.max() > 1:
        covers = covers.clone().detach()/255
    if secrets.max() > 1:
        secrets = secrets.clone().detach()/255
    try:
        _, c, h, w = covers.shape
    except:
        c, h, w = covers.shape
    #
    # Prep images
    #
    cover_input = dwt(covers)
    secret_input = dwt(secrets)
    input_img = torch.cat((cover_input, secret_input), 1)
    #
    # Steg Hide
    #
    with torch.no_grad():
        steg_img, attack_container, output_z, output_container, input_container = embed_attack(net, input_img, "JPEG Q=80")
        #
        # Steg reveal
        #
        output_rev = torch.cat((input_container, output_z), 1)
        output_image = net(output_rev, rev=True)
        extracted = output_image.narrow(1, 4 * 3,
                                         output_image.shape[1] - 4 * 3)
        extracted = iwt(extracted)
        ### ---> maybe need post enhance ?? 
        extracted = net.post_enhance(extracted)

    steg_max = steg_img.max()
    steg_min = steg_img.min()
    extract_max = extracted.max()
    extract_min = extracted.min()
        
    return (steg_img - steg_min) / (steg_max - steg_min), (extracted - extract_min) / (extract_max - extract_min), output_z

def add_gauss(imgs, mu=0, sigma=0.1):
    """
    Add gaussian noise to images.
    
    Parameters
    ----------
    imgs : tensor
        tensor of images
    mu : float
        mean
    sigma : float
        std
    """
    # creat a mask for the noise
    rnd_tnsor = torch.rand(imgs.shape)
    mask = (rnd_tnsor > 0.5).float()
    
    dim = list(imgs.shape)
    n = torch.normal(mu, sigma, dim)
    n = mask*n
    
    return torch.clip(imgs + n, 0, 1)

def _add_dct_noise_batch(images, sigma, ratio_change):
    """
    DCT sanitize on an entire batch of images. Return the batch of sanitized
    images.

    Args:
        images (torch.Tensor) : Images to sanitize
        sigma (float) : Standard deviation of the noise to add to the DCT
        ratio_change (float) : ratio of the total number of frequencies to change. 
                               Between 0 and 1. 
                            
    Returns:
        sanitize_batch (torch.Tensor) : batch of DCT sanitized images 
    """
    sanitize_batch = torch.zeros_like(images)

    for i in range(images.shape[0]):
        sanitize_batch[i] = add_dct_noise(images[i], sigma, ratio_change)

    return sanitize_batch

def add_dct_noise(image, sigma=0.2, ratio_change=0.20):
    """
    DCT sanitize an images. Return the DCT sanitized image.

    Args:
        images (torch.Tensor) : Image to sanitize
        sigma (float) : Standard deviation of the noise to add to the DCT
        ratio_change (float) : ratio of the total number of frequencies to change. 
                               Between 0 and 1. 
                            
    Returns:
        sanitize_batch (torch.Tensor) : DCT sanitized images 
    """
    if len(image.shape) > 3:
        return _add_dct_noise_batch(image, sigma, ratio_change)
    #
    # Convert to numpy
    #
    image_np = np.array(image.permute(1, 2, 0))
    N, M, CH = image_np.shape
    frequencies = int(M*ratio_change)
    #
    # for each RGB, convert to dct and sanitize
    #
    for channel in range(CH):
        layer_dct = fftpack.dct(image_np[:,:,channel], norm='ortho')
        mu = np.mean(layer_dct)
        #
        # add noise
        #
        # noise = np.random.normal(mu, 0.1, size=(N-N//2, M-M//2))
        noise = np.random.normal(mu, sigma, size=(N, frequencies))
        layer_dct[:, M-frequencies:] = noise
        image_np[:, :, channel] = fftpack.idct(layer_dct, norm='ortho')
    #
    # Return torch tensor of image
    #
    return torch.Tensor(image_np).permute(2, 0, 1)

def add_gaussian_noise(images, mean=0., std=0.1):
    """
    Add Gaussian noise to a batch of images.

    :param images: Input images of shape [batch size, channels, h, w]
    :param mean: Mean of the Gaussian noise
    :param std: Standard deviation of the Gaussian noise
    :return: Images with added Gaussian noise
    """
    
    noise = torch.randn_like(images) * std + mean
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)  # Clamping to ensure values are between 0 and 1
