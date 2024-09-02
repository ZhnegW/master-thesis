from fitness_check.data_access import PhysioDataset
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from fitness_check.transforms import filterBank, gammaFilter, MSD, Energy_Wavelet

class cal_thresholds():

    def __init__(self, data, test_data):
        self.data = data.permute(1, 3, 0, 2)  # (channels, frequency, trials, time)
        self.data = self.data.numpy()
        self.data = self.data[0,0,:,:] # (trials, time)
        self.test_data = test_data.permute(1, 3, 0, 2)  # (channels, frequency, trials, time)
        self.test_data = self.test_data.numpy()
        self.test_data = self.test_data[0,0,:,:] # (trials, time)
        # self.data = data.squeeze(1)
        # self.test_data = test_data.squeeze(1)
        self.sig = self.get_sig()
        self.sig_thresholds = self.get_sig_thresholds()

    def get_sig(self):
        run4 = self.data[0:15,:].flatten()
        run8 = self.data[15:30,:].flatten()
        run10 = self.data[30:45,:].flatten()
        run12 = self.data[45:60,:].flatten()
        sig1 = run4[0:800] # the first 5s of run4
        sig2 = run8[800:1600] # the second 5s of run8
        sig3 = run10[1600:2400] # the third 5s of run10
        sig4 = run12[2400:3200] # the fourth 5s of run12
        sig = np.concatenate((sig1,sig2,sig3,sig4),axis=0)

        return sig

    def RMSE(self, x, y):
        mse = mean_squared_error(x, y)
        rmse = math.sqrt(mse)

        return rmse

    def get_sig_thresholds(self):
        sig_thresholds = []
        for i in range(0, self.data.shape[0], 5):
            sig_thresholds.append(self.RMSE(self.sig, self.data[i:i+5,:].flatten()))
        return sig_thresholds

    def get_threshold(self):
        seg_thresholds = np.zeros((12,8))
        for i in range(0, 12):
            seg_thresholds[i,:] = [self.RMSE(self.data[i*5:(i+1)*5,:].flatten(), self.data[5:10,:].flatten()),
                                   self.RMSE(self.data[i*5:(i+1)*5,:].flatten(), self.data[10:15,:].flatten()),
                                   self.RMSE(self.data[i*5:(i+1)*5,:].flatten(), self.data[20:25,:].flatten()),
                                   self.RMSE(self.data[i*5:(i+1)*5,:].flatten(), self.data[25:30,:].flatten()),
                                   self.RMSE(self.data[i*5:(i+1)*5,:].flatten(), self.data[35:40,:].flatten()),
                                   self.RMSE(self.data[i*5:(i+1)*5,:].flatten(), self.data[40:45,:].flatten()),
                                   self.RMSE(self.data[i*5:(i+1)*5,:].flatten(), self.data[50:55,:].flatten()),
                                   self.RMSE(self.data[i*5:(i+1)*5,:].flatten(), self.data[55:60,:].flatten())]

        run4_seg1 = (np.sum(seg_thresholds[0,:]) + self.sig_thresholds[0]) / 9
        run4_seg2 = (np.sum(seg_thresholds[1,1:]) + self.sig_thresholds[1]) / 8
        run4_seg3 = (seg_thresholds[2,0] + np.sum(seg_thresholds[2,2:]) + self.sig_thresholds[2]) / 8
        run8_seg1 = (np.sum(seg_thresholds[3,:]) + self.sig_thresholds[3]) / 9
        run8_seg2 = (np.sum(seg_thresholds[4,0:2]) + np.sum(seg_thresholds[4,3:]) + self.sig_thresholds[4]) / 8
        run8_seg3 = (np.sum(seg_thresholds[5,0:3]) + np.sum(seg_thresholds[5,4:]) + self.sig_thresholds[5]) / 8
        run10_seg1 = (np.sum(seg_thresholds[6,:]) + self.sig_thresholds[6]) / 9
        run10_seg2 = (np.sum(seg_thresholds[7,0:4]) + np.sum(seg_thresholds[7,5:]) + self.sig_thresholds[7]) / 8
        run10_seg3 = (np.sum(seg_thresholds[8,0:5]) + np.sum(seg_thresholds[8,6:]) + self.sig_thresholds[8]) / 8
        run12_seg1 = (np.sum(seg_thresholds[9,:]) + self.sig_thresholds[9]) / 9
        run12_seg2 = (np.sum(seg_thresholds[10,0:6]) + np.sum(seg_thresholds[10,7:]) + self.sig_thresholds[10]) / 8
        run12_seg3 = (np.sum(seg_thresholds[11,0:7]) + self.sig_thresholds[11]) / 8

        run4_threshold = max(run4_seg1, run4_seg2, run4_seg3)
        run8_threshold = max(run8_seg1, run8_seg2, run8_seg3)
        run10_threshold = max(run10_seg1, run10_seg2, run10_seg3)
        run12_threshold = max(run12_seg1, run12_seg2, run12_seg3)

        threshold = np.average([run4_threshold, run8_threshold, run10_threshold, run12_threshold]) - np.std([run4_threshold, run8_threshold, run10_threshold, run12_threshold])

        return threshold

    def get_score(self, index1, index2):
        sig = self.get_sig()
        score = self.RMSE(sig, self.test_data[index1:index2,:].flatten())

        return score

if __name__ == '__main__':
        train_data = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='train', transform=filterBank([[8,13]],160), preprocess=False)
        test_data = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='test', transform=filterBank([[8,13]],160), preprocess=False)
        # print(train_data.getdata().shape) # (60,1,640,1)
        threshold = cal_thresholds(train_data.getdata(), test_data.getdata()).get_threshold()
        print(threshold)
        for i in range(0, 30, 5):
            score = cal_thresholds(train_data.getdata(), test_data.getdata()).get_score(i, i+5)
            print(score)
