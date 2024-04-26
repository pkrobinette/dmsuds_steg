# Sanitizing Hidden Information with Diffusion Models
These experiments were conducted on a macOS 13.5.2 with an Apple M2 Max processor with 64 GB of memory.

## Installation
1. Install all necessary dependencies and folders
```
chmod +x setup.sh && ./setup.sh && conda activate dmsuds
```

2. Download Necessary Models and Datasets
   
   a. Download the following checkpoint from [https://github.com/openai/improved-diffusion] to `models/diffusion_models/cifar10_uncond_50M_500K.pt`:
      > Unconditional CIFAR-10 with our L_hybrid objective and cosine noise schedule (cifar10_uncond_50M_500K.pt)

   b. Download the following checkpoint from [https://github.com/openai/improved-diffusion] to `models/diffusion_models/imagenet64_uncond_100M_1500K.pt`
      > imagenet64_uncond_100M_1500K.pt (Unconditional ImageNet-64 with our L_hybrid objective)

   c. Download the following checkpoint from [https://github.com/openai/improved-diffusion] to `models/diffusion_models/imagenet64_uncond_vlb_100M_1500K.pt`:
      > imagenet64_uncond_vlb_100M_1500K.pt (Unconditional ImageNet-64 with the L_vlb objective and cosine noise schedule)

   d. Download the ImageNet dataset to the following directory:
      > datasets/ImageNet

   e. Download the UrbanSound8K dataset to the following directory:
      > audio_case_study/data/UrbanSound8K


## Artifact Instructions (~15 hrs to run)
1. All models are pre-trained. Reproduce results by running:
```
chmod +x scripts/* && ./scripts/run_all.sh
```
2. All results are saved to the `results` folder. To recreate a specific research question (RQ), see the command map below.

| RQ | Python Script | Result Location |
| -------- | -------- | -------- |
| **RQ1** | `chmod +x scripts/rq1.sh && ./scripts/rq1.sh` | `results/*` |
| **RQ2** | `chmod +x scripts/rq2.sh && ./scripts/rq2.sh` | `results/*` |
| **RQ3** | `chmod +x scripts/rq3.sh && ./scripts/rq3.sh` | `results/*` |
| **RQ4** | `chmod +x scripts/rq4.sh && ./scripts/rq4.sh` | `results/*` |


#### Directories
> `audio_case_study`: files for the audio case study

> `configs`: training config files for udh and ddh

> `models`: all pre-trained models

> `results`: where results are saved

> `scripts`: easy to run training and testing scripts

> `utils`: Helper functions. Model files. Etc.


