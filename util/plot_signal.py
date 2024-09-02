from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import pyedflib
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
# import picard
from scipy.stats import skew, iqr, zscore, kurtosis, entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from util.transforms import filterBank, gammaFilter, MSD
# from transforms import MTF
# from PDC import ChannelSelection
from util import mspca

class PhysioDataset(Dataset):
    def get_subject_data(self, subject, path, No_channels, Window_Length=640, label=1, train='train'):

        mymodel = mspca.MultiscalePCA()

        if train == 'train':
            run = ["04", "06", "08", "10"]
        elif train == 'valid':
            run = ["12"]
        elif train == 'test':
            run = ["14"]

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
            for i in annotation[1]:  # annotation[1]每个task结束的时间点
                marker.append(i * 160)  # 每个task有多少和Datapoints len(marker)=30, 因为一个run有30个trial
            y = []  # y是每个Datapoint的标签，len(y) = len(Datapoints)
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
            totalSignals = file.signals_in_file  # totalSignals = 64, channel number
            signal_labels = file.getSignalLabels()  # label names of electrode in 10-10 system
            trial = 0
            trial_segment = []  # y!=0的trial的开始时的index
            if y[0] != 0:
                trial = 1
                trial_segment.append(0)
            for i in range(1, len(y)):
                if y[i] != y[i - 1] and y[i] != 0:
                    trial = trial + 1
                    trial_segment.append(i)
            data = np.zeros((trial, totalSignals, 640))  # 每个trial持续4s * 采样率160hz = 640
            for i in range(trial):
                data_each_trial = np.zeros((totalSignals, 640))
                for j in np.arange(totalSignals):
                    data_each_trial[j, :] = file.readSignal(j)[trial_segment[i]:trial_segment[i]+640]
                data_mspca = mymodel.fit_transform(data_each_trial, wavelet_func='db4', threshold=0.3)  # 对一个trial所有通道信号降噪
                data[i, :, :] = data_mspca
            data_run.append(data)

        data_return = data_run[0]
        for i in range(1, len(data_run)):
            data_return = np.concatenate((data_return, data_run[i]), axis=0)
        class_return = np.zeros(data_return.shape[0])
        for i in range(class_return.shape[0]):
            class_return[i] = label

        return data_return, class_return

    def __init__(self, subject, path, train='train', transform=None, channels=None, use_channel_no=0):
        self.channel_selected = channels
        self.transform = transform
        self.use_channel_no = use_channel_no
        No_channels = 64

        if train == 'train':
            self.data_return, self.class_return = self.get_subject_data(subject, path, No_channels, label=1,
                                                                        train=train)
            self.data_return = self.data_return[:, self.channel_selected, :]
            for i in [x for x in range(1, 21) if x != subject]:
                negative_data, negative_class = self.get_subject_data(i, path, No_channels, label=0, train=train)
                negative_data = negative_data[:, self.channel_selected, :]
                self.data_return = np.concatenate((self.data_return, negative_data[:5, :, :]), axis=0)  # 只取neg_data的前5个trial，最终增加49*5个trial
                self.class_return = np.concatenate((self.class_return, negative_class[:5]), axis=0)
        elif train == 'valid':
            self.data_return, self.class_return = self.get_subject_data(subject, path, No_channels, label=1,
                                                                        train=train)
            self.data_return = self.data_return[:, self.channel_selected, :]
            for i in [x for x in range(1, 21) if x != subject]:
                negative_data, negative_class = self.get_subject_data(i, path, No_channels, label=0, train=train)
                negative_data = negative_data[:, self.channel_selected, :]
                self.data_return = np.concatenate((self.data_return, negative_data[:5, :, :]),
                                                  axis=0)  # 只取neg_data的前5个trial，最终增加49*5个trial
                self.class_return = np.concatenate((self.class_return, negative_class[:5]), axis=0)
        elif train == 'test':
            self.data_return, self.class_return = self.get_subject_data(subject, path, No_channels, label=1,
                                                                        train=train)
            self.data_return = self.data_return[:, self.channel_selected, :]
            for i in [x for x in range(21, 41) if x != subject and x not in [88,92]]:
                negative_data, negative_class = self.get_subject_data(i, path, No_channels, label=0, train=train)
                negative_data = negative_data[:, self.channel_selected, :]
                self.data_return = np.concatenate((self.data_return, negative_data[:5, :, :]),
                                                  axis=0)  # 只取neg_data的前5个trial，最终增加49*5个trial
                self.class_return = np.concatenate((self.class_return, negative_class[:5]), axis=0)

    def __len__(self):
        return self.data_return.shape[0]

    def __getitem__(self, index):
        data = self.data_return[index, :, :640]
        # self.plot(data[0, :], "raw.png")
        if self.transform:
            data = self.transform(data)
        label = self.class_return[index]
        # self.plot(data[0, :], "filtered.png")

        return data, int(label)

    def plot(self, data, title):
        # data = np.transpose(data, (1,0))
        fig, axs = plt.subplots(2, 1, figsize=(20, 40))
        for i in range(2):
            if title == "ICA_component.png":
                axs[i].plot(data[:, i])
            else:
                axs[i].plot(data)
        fig.savefig(title)


if __name__ == '__main__':
    filterTransform = filterBank([[1,4],[4,8],[8,13],[13,30]], 160)
    # filterTransform = gammaFilter()
    data_transform = transforms.Compose([filterTransform, MSD()])

    train_data = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='valid',
                               transform=filterTransform, channels=[8, 10, 12])
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
    data, classes = next(iter(train_dataloader))
    print(data.shape)
