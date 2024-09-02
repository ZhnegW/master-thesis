from fitness_check.data_access import PhysioDataset
import numpy as np
import math
from fitness_check.transforms import filterBank, gammaFilter, MSD, Energy_Wavelet
import scipy
import torch
import pywt
from qim import QIM
from stn import STN,Ssn

train_data = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='train', transform=None, preprocess=False)
data = train_data.getdata()  # trials, channel 3, time
print(data.shape)
data = data[0]

# ssn = Ssn(10)
# noise_signal = []
# for i in range(data.shape[0]):
#     signal = ssn.generate_snr(data[i])
#     noise_signal.append(signal)
# noise_signal = np.vstack(noise_signal)
# print(np.real(noise_signal))

# DWT
LL, (LH, HL, HH) = pywt.dwt2(data, 'haar')

# SVD
u, s, vh = np.linalg.svd(HH, False)  # (2, 2) (2,) (2, 320)

# qim
qim = QIM(10)
watermarker = np.array([0,1])
watermarked_s = qim.embed(s, watermarker)

# Inverse SVD
HH = np.dot(u*watermarked_s, vh)

# IDWT
coeffs = (LL, (LH, HL, HH))
watermarked_data = pywt.idwt2(coeffs, 'haar')

# extract watermarker
#DWT
LL, (LH, HL, HH) = pywt.dwt2(data, 'haar')
#SVD
u, s, vh = np.linalg.svd(HH, False)
#extract watermarker
HH_detected, watermarker_detected = qim.detect(s)
print(watermarker_detected)

stn = STN(1)
noise_signal = []
print(watermarked_data.shape)
for i in range(watermarked_data.shape[0]):
    signal = stn.generate_snr(watermarked_data[i])
    noise_signal.append(signal)
noise_signal = np.vstack(noise_signal)

# extract watermarker
#DWT
LL, (LH, HL, HH) = pywt.dwt2(noise_signal, 'haar')
#SVD
u, s, vh = np.linalg.svd(HH, False)
#extract watermarker
HH_detected, watermarker_detected = qim.detect(s)
print(watermarker_detected)

