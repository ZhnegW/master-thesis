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

    def __init__(self, num_channels):
        super(VSMSCNN, self).__init__()
        self.num_channels = num_channels

        # Define the first scale convolutional layers
        self.conv1_scale1 = nn.Conv1d(in_channels=num_channels, out_channels=2, kernel_size=1, stride=1, padding='same')
        self.pool1_scale1 = nn.MaxPool1d(kernel_size=1, stride=4, padding=0)

        # Define the second scale convolutional layers
        self.conv1_scale2 = nn.Conv1d(in_channels=num_channels, out_channels=2, kernel_size=3, stride=1, padding='same')
        self.pool1_scale2 = nn.MaxPool1d(kernel_size=3, stride=4, padding=1)

        # Define the third scale convolutional layers
        self.conv1_scale3 = nn.Conv1d(in_channels=num_channels, out_channels=2, kernel_size=5, stride=1, padding='same')
        self.pool1_scale3 = nn.MaxPool1d(kernel_size=5, stride=4, padding=0)

        # Define the final scale convolutional layers
        self.pool1_scale4 = nn.MaxPool1d(kernel_size=3, stride=4, padding=1)
        self.conv1_scale4 = nn.Conv1d(in_channels=num_channels, out_channels=2, kernel_size=3, stride=1, padding='same')

        self.flatten = nn.Flatten()

        # Define the flatten layer

        self.conv2_1 = nn.Conv2d(in_channels=num_channels, out_channels=12, kernel_size=3, stride=1, padding='same')
        self.pool2_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2_2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=1, stride=1, padding='same')
        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_last = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding='same')
        self.maxpool_last = nn.MaxPool1d(kernel_size=3, stride=4, padding=0)

        self.fc1 = torch.nn.Linear(5524, 1024)  # parameters set according to the Physionet dataset
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.output_layer = torch.nn.Linear(256, 2)

    def forward(self, x0, x1):
        # First part of the model
        # print(x0.shape)  # (batch_size, num_channels=3, sample_size=640, 4)
        x0 = x0.permute(3, 0, 1, 2)  # (4, batch_size, 3, sample_size)
        output_1 = torch.tensor([]).to('cuda', dtype=torch.float)
        for i in range(4):  # 4 frequencies
            # First scale  # (batch_size, 2, sample_size/4)
            x0_1 = self.conv1_scale1(x0[i])
            # x1 = nn.functional.relu(x1)
            x0_1 = self.pool1_scale1(x0_1)

            # Second scale
            x0_2 = self.conv1_scale2(x0[i])
            # x2 = nn.functional.relu(x2)
            x0_2 = self.pool1_scale2(x0_2)

            # Third scale
            x0_3 = self.conv1_scale3(x0[i])
            # x3 = nn.functional.relu(x3)
            x0_3 = self.pool1_scale2(x0_3)

            # Final scale
            x0_4 = self.pool1_scale4(x0[i])
            x0_4 = self.conv1_scale2(x0_4)

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
                x_stft = self.Stft(x_vmd)   # (80, 180)
                x_stft = np.expand_dims(x_stft, axis=0)  # (1, 80, 180)
                X_list.append(x_stft)
            X = np.vstack(X_list)  # (3, 80, 180)
            X = np.expand_dims(X, axis=0)  # (1, 3, 84, 180)
            output_vmdstft.append(X)
        output_vmdstft = np.vstack(output_vmdstft)  # (batch_size, 3, 80, 180)

        input = torch.from_numpy(output_vmdstft).float()
        input = input.to('cuda', dtype=torch.float)
        output_2 = self.conv2_1(input)
        output_2 = self.pool2_1(output_2)
        output_2 = self.conv2_2(output_2)
        output_2 = self.pool2_2(output_2)
        output_2 = self.flatten(output_2)

        # Concatenate the two parts
        output = torch.cat((output_1, output_2), dim=1)  # (batch_size, 2*4*sample_size+12*40*90)
        output = output.unsqueeze(1)  # (batch_size, 1, 2*4*sample_size+12*40*90)
        output = self.conv_last(output)
        output = self.maxpool_last(output)  # (batch_size, 2, 2762)
        output = self.flatten(output)  # (batch_size, 2*2762)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.output_layer(output)

        return output


if __name__ == '__main__':
    from torchinfo import summary
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