from utils.classifier import Classifier
import torch.nn as nn
from data_imagenet import ImageNetDataset
import torch
import matplotlib.pyplot as plt
from utils.StegoPy import encode_img, decode_img
from utils.DSM_imagenet import DiffusionSanitizationModel
from tqdm import tqdm


def main():
    #
    # Load dataset and models
    #
    print("\nLoading datasets and models ...............................\n")
    dataset = ImageNetDataset("datasets/ImageNet", "test", num_images=500, imsize=128)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False, pin_memory=True)
    model = torch.load("supp_detect_model.pth")
    diff_model = DiffusionSanitizationModel("models/diffusion_models/imagenet64_uncond_100M_1500K.pt")
    #
    # test
    #
    print("\nTesting ...............................\n")
    model.eval()
    total_corr = 0
    total_num = 0
    for i, data in enumerate(tqdm(data_loader)):
        num = data.shape[0] // 2
        secret = data[:num]
        cover = data[num:num * 2]
        #
        # Make containers and difference
        #
        containers = encode_img(cover*255, secret*255, train_mode=True)/255
        covers_out = diff_model.sanitize(cover)
        containers_out = diff_model.sanitize(containers)
        diff_covers = covers_out - cover
        diff_containers = containers_out - containers
        #
        # Make labels and image set
        #
        labels = torch.cat((torch.zeros(cover.shape[0]), torch.ones(containers.shape[0]))).long()
        images = torch.cat((diff_covers, diff_containers))
        #
        # test classifier
        #
        with torch.no_grad():
            out = model(images)
            corr = sum([1 if o == t else 0 for o, t in zip(out.argmax(dim=1), labels)])
            total_num += data.shape[0]
            total_corr += corr
            
    print(f"Accuracy: {total_corr/total_num}")

if __name__ == "__main__":
    main()