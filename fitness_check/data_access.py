from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import pyedflib
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from picard import picard
from scipy.stats import skew, iqr, zscore, kurtosis, entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from fitness_check.transforms import filterBank, gammaFilter, MSD, Energy_Wavelet


class PhysioDataset(Dataset):
    def get_subject_data(self, subject, path, train='train'):
        if train == 'train':
            run = ["04", "06", "08", "10"]
        else:
            run = ["12", "14"]

        data_run = []

        for run_num in run:
            if (subject < 10):
                file = pyedflib.EdfReader(
                    path + "/S00" + str(subject) + "/" + "S00" + str(subject) + "R" + run_num + ".edf")
            else:
                file = pyedflib.EdfReader(
                    path + "/S0" + str(subject) + "/" + "S0" + str(subject) + "R" + run_num + ".edf")
            annotation = file.readAnnotations()
            marker = []
            for i in annotation[1]:
                marker.append(i * 160)
            y = []
            for counter, dataPoints in enumerate(marker):
                for i in range(int(dataPoints)):
                    code = annotation[2][counter]
                    if code == 'T0':
                        y.append(0)
                    elif code == 'T1':
                        y.append(1)
                    elif code == 'T2':
                        y.append(2)
                    else:
                        # TODO
                        print("catch error here")
            totalSignals = file.signals_in_file  # totalSignals = 64
            signal_labels = file.getSignalLabels()  # label names of electrode in 10-10 system
            trial = 0
            trial_segment = []
            if y[0] != 0:
                trial = 1
                trial_segment.append(0)
            for i in range(1, len(y)):
                if y[i] != y[i - 1] and y[i] != 0:
                    trial = trial + 1
                    trial_segment.append(i)
            data = np.zeros((trial, totalSignals, 640))
            for i in range(trial):
                for j in np.arange(totalSignals):
                    data[i, j, :] = file.readSignal(j)[trial_segment[i]:trial_segment[i] + 640]
                # data[i, :, :] = (data[i, :, :] - np.mean(data[i, :, :])) / np.std(data[i, :, :])
            data_run.append(data)

        data_return = data_run[0]
        for i in range(1, len(data_run)):
            data_return = np.concatenate((data_return, data_run[i]), axis=0)

        return data_return

    def __init__(self, subject, path, train='train', transform=None, preprocess=False,
                 channels=[1, 2, 4, 5, 8, 9, 11, 12, 15, 16, 18, 19]):

        self.channel_selected = channels
        self.transform = transform

        if train == 'train':
            self.data_return = self.get_subject_data(subject, path, train='train')
            if preprocess:
                self.data_return = self.KMAR_PIC(self.data_return)
            self.data_return = self.data_return[:, self.channel_selected, :]
            if transform is not None:
                self.data_return = self.transform(self.data_return)
        elif train == 'test':
            self.data_return = self.get_subject_data(subject, path, train='test')
            if preprocess:
                self.data_return = self.KMAR_PIC(self.data_return)
            self.data_return = self.data_return[:, self.channel_selected, :]
            if transform is not None:
                self.data_return = self.transform(self.data_return)

    def KMAR(self, data):
        for i in range(data.shape[0]):
            data[i, :, :] = zscore(data[i, :, :])
        data = np.transpose(data, (1, 0, 2))
        data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
        # K, W, Y = picard(data)
        data = np.transpose(data, (1, 0))
        ica = FastICA(n_components=data.shape[1])
        ica.fit(data)
        Y = ica.fit_transform(data)
        # self.plot(Y, "ICA_component.png")
        Y_var = np.var(Y, axis=0, keepdims=True)
        Y_skew = skew(Y, axis=0).reshape((-1, Y.shape[1]))
        Y_iqr = iqr(Y, axis=0, interpolation='midpoint', keepdims=True)
        features = np.concatenate((Y_var, Y_skew, Y_iqr), axis=0)
        features = np.transpose(features, (1, 0))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
        artif_label = np.bincount(kmeans.labels_).argmin()
        for i, label in enumerate(kmeans.labels_):
            if (label == artif_label):
                Y[:, i] = 0
        data = ica.inverse_transform(Y)
        data = np.transpose(data, (1, 0))
        data = np.reshape(data, (data.shape[0], data.shape[1] // 640, 640))
        data = np.transpose(data, (1, 0, 2))
        return data

    def KMAR_PIC(self, data):
        for i in range(data.shape[0]):
            data[i, :, :] = zscore(data[i, :, :])
        data = np.transpose(data, (1, 0, 2))
        data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
        K, W, Y = picard(data, max_iter=150, tol=1e-06)
        Y_ptp = np.ptp(Y, axis=1, keepdims=True)
        Y_max = np.amax(Y, axis=1, keepdims=True)
        # var
        Y_var4 = np.var(Y, axis=1, keepdims=True)
        # local var
        local_var = np.zeros((Y.shape[0], Y.shape[1] // 160))
        for i in range(0, Y.shape[1] // 160 - 1):
            local_var[:, i] = np.var(Y[:, i * 160:(i + 1) * 160], axis=1)
        Y_var = np.mean(local_var, axis=1, keepdims=True)
        local_var15 = np.zeros((Y.shape[0], Y.shape[1] // 40))
        for i in range(0, Y.shape[1] // 40 - 1):
            local_var15[:, i] = np.var(Y[:, i * 40:(i + 1) * 40], axis=1)
        Y_var15 = np.mean(local_var15, axis=1, keepdims=True)
        # local_skew
        local_skew = np.zeros((Y.shape[0], Y.shape[1] // 160))
        for i in range(0, Y.shape[1] // 160 - 1):
            local_skew[:, i] = skew(Y[:, i * 160:(i + 1) * 160], axis=1)
        Y_skew = np.mean(local_skew, axis=1, keepdims=True)
        local_skew15 = np.zeros((Y.shape[0], Y.shape[1] // 40))
        for i in range(0, Y.shape[1] // 40 - 1):
            local_skew15[:, i] = skew(Y[:, i * 40:(i + 1) * 40], axis=1)
        Y_skew15 = np.mean(local_skew15, axis=1, keepdims=True)
        # #local_iqr
        # local_iqr = np.zeros((Y.shape[0],Y.shape[1]//160))
        # for i in range(0, Y.shape[1]//160 - 1):
        #     local_iqr[:,i] = iqr(Y[:, i*160:(i+1)*160], axis=1, interpolation='midpoint')
        # Y_iqr = np.mean(local_iqr, axis=1, keepdims=True)
        # kurtosis
        Y_kur = kurtosis(Y, axis=1).reshape((Y.shape[0], -1))
        # Max First Derivative
        Y_der = np.amax(np.gradient(Y, axis=1), axis=1, keepdims=True)
        # entropy
        Y_copy = np.around(Y, decimals=4)
        Y_ent = np.zeros((Y.shape[0], 1))
        for i in range(Y_copy.shape[0]):
            values, counts = np.unique(Y_copy[i, :], return_counts=True)
            Y_ent[i, :] = entropy(counts)
        # variance of variance
        Y_varvar = np.var(local_var, axis=1, keepdims=True)
        Y_varvar15 = np.var(local_var15, axis=1, keepdims=True)
        features = np.concatenate(
            (Y_var4, Y_max, Y_ptp, Y_kur, Y_der, Y_ent, Y_var, Y_var15, Y_skew, Y_skew15, Y_varvar, Y_varvar15), axis=1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
        artif_label = np.bincount(kmeans.labels_).argmin()
        # artif_label = kmeans.labels_[np.argmax(Y_var)]
        remove_1 = 0
        max_var = 0
        for i, label in enumerate(kmeans.labels_):
            if label == artif_label and Y_var15[i, 0] > max_var:
                remove_1 = i
                max_var = Y_var15[i, 0]
        remove_2 = 0
        max_var = 0
        for i, label in enumerate(kmeans.labels_):
            if label == artif_label and i != remove_1 and Y_var4[i, 0] > max_var:
                remove_2 = i
                max_var = Y_var4[i, 0]
        for i, label in enumerate(kmeans.labels_):
            if i == remove_1 or i == remove_2:
                Y[i, :] = 0
        # removed = np.argpartition(Y_var, -2, axis=0)[-2:]
        # for i in removed:
        #     Y[i, :] = 0
        W_inv = np.linalg.inv(W)
        K_inv = np.linalg.inv(K)
        data = K_inv.dot(W_inv).dot(Y)
        data = np.reshape(data, (data.shape[0], data.shape[1] // 640, 640))
        data = np.transpose(data, (1, 0, 2))
        return data

    def getdata(self):
        data = self.data_return
        return data

    # def __len__(self):
    #     return self.data_return.shape[0]
    #
    # def __getitem__(self, index):
    #     data = self.data_return[index]
    #     if self.transform:
    #         data = self.transform(data)
    #
    #     return data


if __name__ == '__main__':
    # filterTransform = filterBank([[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]], 160)
    filterTransform = filterBank([[8, 13]], 160)

    train_data = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='train',
                               transform=filterTransform, preprocess=False)
    # train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    # data = next(iter(train_dataloader))
    print(train_data.getdata().shape)