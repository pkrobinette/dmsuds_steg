# Monsters in the Dark: Sanitizing Hidden Threats with Diffusion Models
These experiments were conducted on a macOS 13.5.2 with an Apple M2 Max processor with 64 GB of memory.

## Installation

### Conda Environment (Recommended)
This creates, activates, and installs all necessary dependencies.

```
conda create -y -n dmsuds pip python=3.8 && conda activate dmsuds && pip install -r requirements.txt
```

### Download Necessary Models and Datasets
1. Download the following checkpoint from [https://github.com/openai/improved-diffusion] to `models/diffusion_models/cifar10_uncond_50M_500K.pt`:
   > Unconditional CIFAR-10 with our L_hybrid objective and cosine noise schedule (cifar10_uncond_50M_500K.pt)
2. Download the following checkpoint from [https://github.com/openai/guided-diffusion] to `models/diffusion_models/256x256_diffusion_uncond.pt`:
   > 256x256_diffusion_uncond.pt
3. Download the following checkpoint from [https://github.com/openai/improved-diffusion] to `models/diffusion_models/imagenet64_uncond_100M_1500K.pt`
   > imagenet64_uncond_100M_1500K.pt (Unconditional ImageNet-64 with our L_hybrid objective)
4. Download the ImageNet dataset to the following directory:
   > datasets/ImageNet


## Artifact Instructions (~1.5 hrs to run)
All models are pre-trained. Reproduce results by:

```
chmod +x scripts/*
./scripts/run_all.sh
```
2. If you would like to reproduce a specific figure, see the index below and run:
```
python *.py
```

## Results Index
All results are saved to the `results` folder. 

| Artifact | Python Script | Result Location |
| -------- | -------- | -------- |
| **Table 1, Figure 2:** | `python exp_sanitize.py && python make_sanitize_picture.py` | `results/*` |
| **Figure 3:** | `python exp_diffusion_steps.py && make_timestep_picture.py` | `results/*` |
| **Figure 4:** | `python exp_noise_effect.py && make_no_noise_picture.py` | `results/*` |
| **Table 2, Figure 5:** |  `python exp_sanitize_imagenet_ddh.py && python exp_sanitize_imagenet_pris.py && make_imagenet_picture.py` | `results/*` |


#### Directories
> `configs`: training config files for udh and ddh

> `models`: all pre-trained models

> `results`: where results are saved

> `scripts`: easy to run training and testing scripts

> `utils`: Helper functions. Model files. Etc.


