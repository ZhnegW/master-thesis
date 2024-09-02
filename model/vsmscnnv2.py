# multiscale CNN model

import torch
import torch.nn as nn
import numpy as np
from vmdpy import VMD
from scipy import signal
from util.data_loader import BCI2aDataset, PhysioDataset
from torch.utils.data import DataLoader

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class VSMSCNN(nn.Module):

    def Vmd(self, x, alpha, tau, K, DC, init, tol):
        u, u_hat, omega = VMD(x, alpha, tau, K, DC, init, tol)
        return u

    def Stft(self, x):
        '''
        output shape: (80, 108)
        the value of parameters are set according to the Physionet dataset
        the input shape is (4, 640)
        row = 20
        col = 108
        '''
        for i in range(4):
            data = x[i, :]
            window = signal.windows.gaussian(32, std=0.75)
            f, t, Zxx = signal.stft(data, fs=160, window=window, nperseg=32, noverlap=26, nfft=38, scaling='psd')
            row = f.shape[0]
            col = t.shape[0]
            result = np.zeros((row * 4, col))
            result[i * 20:(i + 1) * 20, :] = np.abs(Zxx) ** 5
        return result

    def SCALE1(self, num_channels):
        return nn.Sequential(
                nn.Conv1d(in_channels=num_channels, out_channels=12, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm1d(12),
                swish(),
                nn.MaxPool1d(kernel_size=1, stride=4, padding=0)
                )

    def SCALE2(self, num_channels):
        return nn.Sequential(
                nn.Conv1d(in_channels=num_channels, out_channels=12, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm1d(12),
                swish(),
                nn.MaxPool1d(kernel_size=3, stride=4, padding=1)
                )

    def SCALE3(self, num_channels):
        return nn.Sequential(
                nn.Conv1d(in_channels=num_channels, out_channels=12, kernel_size=5, stride=1, padding='same'),
                nn.BatchNorm1d(12),
                swish(),
                nn.MaxPool1d(kernel_size=5, stride=4, padding=2)
                )

    def SCALE4(self, num_channels):
        return nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=4, padding=1),
                nn.Conv1d(in_channels=num_channels, out_channels=12, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm1d(12),
                swish()
                )

    def BLOCK2(self, num_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels=num_channels, out_channels=24, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(24),
                swish(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(12),
                swish(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                )

    def LastConv(self):
        return nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm1d(12),
                swish(),
                nn.AdaptiveAvgPool1d(2048)
                )

    def LastLayer(self):
        return nn.Sequential(
                nn.Linear(2048*12, 10000),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(10000, 5000),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(5000, 1024),  # parameters set according to the Physionet dataset
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
                )

    def __init__(self, num_channels):
        super(VSMSCNN, self).__init__()
        self.num_channels = num_channels

        # Define the first scale convolutional layers
        self.scale1 = self.SCALE1(num_channels)

        # Define the second scale convolutional layers
        self.scale2 = self.SCALE2(num_channels)

        # Define the third scale convolutional layers
        self.scale3 = self.SCALE3(num_channels)

        # Define the final scale convolutional layers
        self.scale4 = self.SCALE4(num_channels)

        self.flatten = nn.Flatten()

        self.block2 = self.BLOCK2(num_channels)

        self.last_conv = self.LastConv()

        self.last_layer = self.LastLayer()

    def forward(self, x0, x1):
        # First part of the model
        # print(x0.shape)  # (batch_size, num_channels=3, sample_size=640, 4)
        x0 = x0.permute(3, 0, 1, 2)  # (4, batch_size, 3, sample_size)
        output_1 = torch.tensor([]).to('cuda', dtype=torch.float)
        for i in range(4):  # 4 frequencies
            # First scale  # (batch_size, 2, sample_size/4)
            x0_1 = self.scale1(x0[i])

            # Second scale
            x0_2 = self.scale2(x0[i])

            # Third scale
            x0_3 = self.scale3(x0[i])

            # Final scale
            x0_4 = self.scale4(x0[i])

            # Concatenate the two scales
            x = torch.cat((x0_1, x0_2, x0_3, x0_4), dim=2)  # (batch_size, 2, 640)
            x = x.unsqueeze(0)  # (1, batch_size, 2, sample_size) # add one dimension
            output_1 = torch.cat((output_1, x), dim=0)  # the MSCNN result of all bands (4, batch_size, 2, sample_size)

        output_1 = output_1.permute(1, 2, 0, 3)  # (batch_size, 2, 4, sample_size)

        # Flatten the tensor
        output_1 = self.flatten(output_1)  # (batch_size, 2*4*sample_size)

        # Second part of the model
        x1 = x1.cpu()  # (batch_size, num_channels=3, sample_size=640)
        x1 = x1.numpy()
        output_vmdstft = []
        for i in range(x1.shape[0]):  # batch_size (3, 640)
            X_list =[]
            for j in range(x1.shape[1]):  # each channel (640,)
                x_vmd = self.Vmd(x1[i,j,:], alpha=2000, tau=0, K=4, DC=0, init=1, tol=1e-7)  # output_size (4,640)
                x_stft = self.Stft(x_vmd)   # (80, 108)
                x_stft = np.expand_dims(x_stft, axis=0)  # (1, 80, 108)
                X_list.append(x_stft)
            X = np.vstack(X_list)  # (3, 80, 108)
            X = np.expand_dims(X, axis=0)  # (1, 3, 84, 108)
            output_vmdstft.append(X)
        output_vmdstft = np.vstack(output_vmdstft)  # (batch_size, 3, 80, 108)

        input = torch.from_numpy(output_vmdstft).float()
        input = input.to('cuda', dtype=torch.float)
        output_2 = self.block2(input)
        output_2 = self.flatten(output_2)

        # Concatenate the two parts
        output = torch.cat((output_1, output_2), dim=1)  # (batch_size, 2*4*sample_size+12*40*90)
        output = output.unsqueeze(1)  # (batch_size, 1, 2*4*sample_size+12*40*90)
        output = self.last_conv(output)
        output = self.flatten(output)  # (batch_size, 2*2762)
        output = self.last_layer(output)

        return output


if __name__ == '__main__':
    from util.transforms import filterBank, gammaFilter, MSD

    filterTransform = filterBank([[1, 4], [4, 8], [8, 13], [13, 30]], 160)
    train_data_1 = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='train', transform=filterTransform,
                               channels=[7, 10, 12])
    train_dataloader_1 = DataLoader(train_data_1, batch_size=len(train_data_1), shuffle=True)

    train_data_2 = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='train', transform=None,
                               channels=[7, 10, 12])
    train_dataloader_2 = DataLoader(train_data_2, batch_size=len(train_data_2), shuffle=True)
    mscnn = VSMSCNN(num_channels=3)
    # size1 = (60, 3, 640, 4)
    # size2 = (60, 3, 640)
    # summary(mscnn, [size1, size2])

    for i, data in enumerate(zip(train_dataloader_1, train_dataloader_2)):
        # input_1 = torch.randn((60, 3, 640, 4))
        # input_2 = torch.randn((60, 3, 640))
        input_1, class_1 = data[0]
        input_2, class_2 = data[1]
        output = mscnn(input_1, input_2)
        print(output.shape)