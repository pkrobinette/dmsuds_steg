from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision
import torch
import random
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
import os
import glob
from PIL import Image

random.seed(10)

class ImageNetDataset(Dataset):
    def __init__(self, 
                path, 
                mode,
                num_images=None,
                imsize=128,
            ):
        self.transforms = T.Compose(
            [T.Resize((imsize, imsize), antialias=None),
             T.ToTensor()
            ]
        )
        self.imsize = imsize
        #
        # Init dataset, triggers, and responses
        #
        self.dataset = glob.glob(path+"/**")
        self.dataset = sorted(self.dataset)
        
        if num_images is not None:
            num_images = min(num_images, len(self.dataset))
            random.shuffle(self.dataset)
            self.dataset = self.dataset[:num_images]

    def __len__(self):
        """
        Get length of dataset
        """
        return len(self.dataset)


    def __getitem__(self, idx):
        """
        Get image, trigger, and trigger response
        """
        image_name = self.dataset[idx]
        image = Image.open(image_name)

        if self.transforms:
            image = self.transforms(image)
        
        return image
        