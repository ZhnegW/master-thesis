from fitness_check.data_access import PhysioDataset
import numpy as np
import math
from fitness_check.transforms import filterBank, gammaFilter, MSD, Energy_Wavelet
import scipy
import torch
import pywt
from qim import QIM
from stn import STN,Ssn
import torch
from DDPM.ddpm_main.denoising_diffusion_pytorch import Unet, GaussianDiffusion


train_data = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='train', transform=None, preprocess=False)
data = train_data.getdata()  # trials, channel 3, time

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True,
    channels=60
)

diffusion = GaussianDiffusion(
    model,
    image_size=(12, 640),
    timesteps=1000    # number of steps
)

training_images = torch.rand(60, 1, 12, 640) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)