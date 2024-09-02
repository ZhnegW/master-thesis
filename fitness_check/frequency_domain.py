# mean, skewness, kurtosis, std, entropy, range of given signal

from fitness_check.data_access import PhysioDataset
import numpy as np
import math
from fitness_check.transforms import filterBank, gammaFilter, MSD, Energy_Wavelet
import scipy
import torch
from statsmodels.tsa.ar_model import AutoReg
import pywt
from sklearn.metrics import mean_squared_error


def shannonEntropy(data, bin_min=-200, bin_max=200, binWidth=2):
    counts, binCenters = np.histogram(data, bins=np.arange(bin_min + 1, bin_max, binWidth))
    nz = counts > 0
    prob = counts[nz] / np.sum(counts[nz])
    entropy = -np.dot(prob, np.log2(prob / binWidth))
    return entropy

def sa(data):
    sa_list = np.zeros(6)
    mean = np.mean(data)
    skewness = scipy.stats.skew(data, nan_policy='raise')
    kurtosis = scipy.stats.kurtosis(data, nan_policy='raise')
    std = np.std(data)
    entropy = shannonEntropy(data)
    range = np.max(data) - np.min(data)
    sa_list = [mean, skewness, kurtosis, std, entropy, range]

    return sa_list

def ar(data):
    model = AutoReg(data, 10, old_names=False)
    res = model.fit()
    ar_coef = res.params
    return ar_coef

def fft(data):
    data = data
    fft = scipy.fft.fft(data, n=120)
    fft = np.abs(fft)
    return fft

def psd(data):
    f, psd = scipy.signal.periodogram(data, fs=160, window='hamming', scaling='density')
    return psd

def dwt(data):
    ca, cd4, cd3, cd2, cd1 = pywt.wavedec(data, 'db1', level=4)
    return ca

def NRMSD(x, y):
    mse = mean_squared_error(x, y)
    rmse = math.sqrt(mse)

    return rmse

if __name__ == '__main__':

    train_data = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='train', transform=filterBank([[8,13]],160), preprocess=False)
    test_data = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='test', transform=filterBank([[8,13]],160), preprocess=False)

    data = train_data.getdata()
    data = torch.permute(data, ([1, 3, 0, 2]))  # (channels, frequency, trials, time)
    data = data.numpy()
    data = data[0,0,:,:] # (60, 640)
    run4_seg1 = data[0:5,:].flatten()
    run4_seg2 = data[5:10,:].flatten()
    run4_seg3 = data[10:15,:].flatten()
    run8_seg1 = data[15:20,:].flatten()
    run8_seg2 = data[20:25,:].flatten()
    run8_seg3 = data[25:30,:].flatten()
    run10_seg1 = data[30:35,:].flatten()
    run10_seg2 = data[35:40,:].flatten()
    run10_seg3 = data[40:45,:].flatten()
    run12_seg1 = data[45:50,:].flatten()
    run12_seg2 = data[50:55,:].flatten()
    run12_seg3 = data[55:60,:].flatten()

    test_data = test_data.getdata()
    test_data = torch.permute(test_data, ([1, 3, 0, 2]))  # (channels, frequency, trials, time)
    test_data = test_data.numpy()
    test_data = test_data[0,0,:,:] # (30, 640)
    test_seg1 = test_data[0:5,:].flatten()
    test_seg2 = test_data[5:10,:].flatten()
    test_seg3 = test_data[10:15,:].flatten()
    test_seg4 = test_data[15:20,:].flatten()
    test_seg5 = test_data[20:25,:].flatten()
    test_seg6 = test_data[25:30,:].flatten()

    data_list = [run4_seg1, run4_seg2, run4_seg3, run8_seg1, run8_seg2, run8_seg3, run10_seg1, run10_seg2, run10_seg3, run12_seg1, run12_seg2, run12_seg3]
    test_data_list = [test_seg1, test_seg2, test_seg3, test_seg4, test_seg5, test_seg6]

    sig1 = run4_seg1[0:800] # the first 5s of run4
    sig2 = run8_seg1[800:1600] # the second 5s of run8
    sig3 = run10_seg1[1600:2400] # the third 5s of run10
    sig4 = run12_seg1[2400:3200] # the fourth 5s of run12
    sig = np.concatenate((sig1,sig2,sig3,sig4),axis=0)
    sig_sa = sa(sig)  # 6
    sig_ar = ar(sig)  # 11
    sig_fft = fft(sig)  # 120
    sig_psd = psd(sig)  # 1601
    # print(sig_psd.shape)
    sig_dwt = dwt(sig)  # 200

    # calculate the nrmsd of SA features
    train_sa = np.zeros((12,6))
    test_sa = np.zeros((6,6))
    for i in range(12):
        train_sa[i,:] = sa(data_list[i])

    for i in range(6):
        test_sa[i,:] = sa(test_data_list[i])

    # for i in range(12):
    #     print(NRMSD(train_sa[i], sig_sa))
    # print('\n')
    #
    # for i in range(6):
    #     print(NRMSD(test_sa[i], sig_sa))

    # calculate the nrmsd of ar features
    train_ar = np.zeros((12, 11))
    test_ar = np.zeros((6, 11))
    for i in range(12):
        train_ar[i, :] = ar(data_list[i])

    for i in range(6):
        test_ar[i, :] = ar(test_data_list[i])

    # for i in range(12):
    #     print(NRMSD(train_ar[i], sig_ar))
    # print('\n')
    #
    # for i in range(6):
    #     print(NRMSD(test_ar[i], sig_ar))

    # calculate the nrmsd of fft features
    train_fft = np.zeros((12, 120))
    test_fft = np.zeros((6, 120))
    for i in range(12):
        train_fft[i, :] = fft(data_list[i])

    for i in range(6):
        test_fft[i, :] = fft(test_data_list[i])

    # for i in range(12):
    #     print(NRMSD(train_fft[i], sig_fft))
    # print('\n')
    #
    # for i in range(6):
    #     print(NRMSD(test_fft[i], sig_fft))

    # calculate the nrmsd of psd features
    train_psd = np.zeros((12, 1601))
    test_psd = np.zeros((6, 1601))
    for i in range(12):
        train_psd[i, :] = psd(data_list[i])

    for i in range(6):
        test_psd[i, :] = psd(test_data_list[i])

    # for i in range(12):
    #     print(NRMSD(train_psd[i], sig_psd))
    # print('\n')
    #
    # for i in range(6):
    #     print(NRMSD(test_psd[i], sig_psd))

    # calculate the nrmsd of dwt features
    train_dwt = np.zeros((12, 200))
    test_dwt = np.zeros((6, 200))
    for i in range(12):
        train_dwt[i, :] = dwt(data_list[i])

    for i in range(6):
        test_dwt[i, :] = dwt(test_data_list[i])

    for i in range(12):
        print(NRMSD(train_dwt[i], sig_dwt))
    print('\n')

    for i in range(6):
        print(NRMSD(test_dwt[i], sig_dwt))