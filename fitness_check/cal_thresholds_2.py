from fitness_check.data_access import PhysioDataset
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from fitness_check.transforms import filterBank, gammaFilter, MSD, Energy_Wavelet
from scipy import signal
import torch

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

# run4_seg1 = (run4_seg1-np.mean(run4_seg1)) / np.std(run4_seg1)
# run4_seg2 = (run4_seg2-np.mean(run4_seg2)) / np.std(run4_seg2)
# run4_seg3 = (run4_seg3-np.mean(run4_seg3)) / np.std(run4_seg3)
# run8_seg1 = (run8_seg1-np.mean(run8_seg1)) / np.std(run8_seg1)
# run8_seg2 = (run8_seg2-np.mean(run8_seg2)) / np.std(run8_seg2)
# run8_seg3 = (run8_seg3-np.mean(run8_seg3)) / np.std(run8_seg3)
# run10_seg1 = (run10_seg1-np.mean(run10_seg1)) / np.std(run10_seg1)
# run10_seg2 = (run10_seg2-np.mean(run10_seg2)) / np.std(run10_seg2)
# run10_seg3 = (run10_seg3-np.mean(run10_seg3)) / np.std(run10_seg3)
# run12_seg1 = (run12_seg1-np.mean(run12_seg1)) / np.std(run12_seg1)
# run12_seg2 = (run12_seg2-np.mean(run12_seg2)) / np.std(run12_seg2)
# run12_seg3 = (run12_seg3-np.mean(run12_seg3)) / np.std(run12_seg3)

data_list = [run4_seg1, run4_seg2, run4_seg3, run8_seg1, run8_seg2, run8_seg3, run10_seg1, run10_seg2, run10_seg3, run12_seg1, run12_seg2, run12_seg3]

sig1 = run4_seg1[0:800] # the first 5s of run4
sig2 = run8_seg1[800:1600] # the second 5s of run8
sig3 = run10_seg1[1600:2400] # the third 5s of run10
sig4 = run12_seg1[2400:3200] # the fourth 5s of run12
sig = np.concatenate((sig1,sig2,sig3,sig4),axis=0)


def RMSE(x, y):
    mse = mean_squared_error(x, y)
    rmse = math.sqrt(mse)

    return rmse

scores_sig = []

score_4_1 = RMSE(sig, run4_seg1)
score_4_2 = RMSE(sig, run4_seg2)
score_4_3 = RMSE(sig, run4_seg3)
score_8_1 = RMSE(sig, run8_seg1)
score_8_2 = RMSE(sig, run8_seg2)
score_8_3 = RMSE(sig, run8_seg3)
score_10_1 = RMSE(sig, run10_seg1)
score_10_2 = RMSE(sig, run10_seg2)
score_10_3 = RMSE(sig, run10_seg3)
score_12_1 = RMSE(sig, run12_seg1)
score_12_2 = RMSE(sig, run12_seg2)
score_12_3 = RMSE(sig, run12_seg3)
print (score_4_1, score_4_2, score_4_3, score_8_1, score_8_2, score_8_3,
       score_10_1, score_10_2, score_10_3, score_12_1, score_12_2, score_12_3, '\n')

for i in range(0,12):
    score_1 = RMSE(data_list[i], data_list[1])
    score_2 = RMSE(data_list[i], data_list[2])
    score_3 = RMSE(data_list[i], data_list[4])
    score_4 = RMSE(data_list[i], data_list[5])
    score_5 = RMSE(data_list[i], data_list[7])
    score_6 = RMSE(data_list[i], data_list[8])
    score_7 = RMSE(data_list[i], data_list[10])
    score_8 = RMSE(data_list[i], data_list[11])
    print(score_1, score_2, score_3, score_4, score_5, score_6, score_7, score_8, '\n')

# test_data = torch.permute(test_data.getdata(), ([1, 3, 0, 2]))  # (channels, frequency, trials, time)
test_data = test_data.getdata().squeeze(dim=1)
test_data = test_data.numpy()
# test_data = test_data[0,0,:,:] # (30, 640)
test_seg1 = test_data[0:5,:].flatten()
test_seg2 = test_data[5:10,:].flatten()
test_seg3 = test_data[10:15,:].flatten()
test_seg4 = test_data[15:20,:].flatten()
test_seg5 = test_data[20:25,:].flatten()
test_seg6 = test_data[25:30,:].flatten()

# test_seg1 = (test_seg1-np.mean(test_seg1)) / np.std(test_seg1)
# test_seg2 = (test_seg2-np.mean(test_seg2)) / np.std(test_seg2)
# test_seg3 = (test_seg3-np.mean(test_seg3)) / np.std(test_seg3)
# test_seg4 = (test_seg4-np.mean(test_seg4)) / np.std(test_seg4)
# test_seg5 = (test_seg5-np.mean(test_seg5)) / np.std(test_seg5)
# test_seg6 = (test_seg6-np.mean(test_seg6)) / np.std(test_seg6)

score_1 = RMSE(test_seg1, sig)
score_2 = RMSE(test_seg2, sig)
score_3 = RMSE(test_seg3, sig)
score_4 = RMSE(test_seg4, sig)
score_5 = RMSE(test_seg5, sig)
score_6 = RMSE(test_seg6, sig)
print(score_1, score_2, score_3, score_4, score_5, score_6)
