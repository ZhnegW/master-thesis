from util.data_loader import BCI2aDataset, PhysioDataset
from torch.utils.data import DataLoader
from vmdpy import VMD

train_data = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='train', transform=None, channels=[8, 10, 12])
train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
data, classes = next(iter(train_dataloader))  # data[60, 3 , 640]
data = data.numpy()
data = data[0, :, :]  # data(640,)

#VMD parameters
# alpha = 2000       # moderate bandwidth constraint
# tau = 0            # noise-tolerance (no strict fidelity enforcement)
# K = 4              # 4 modes
# DC = 0             # no DC part imposed
# init = 1           # initialize omegas uniformly
# tol = 1e-7

#. Run VMD
# u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)  # u(4, 640)
# print(u.shape)
# #. Visualize decomposed modes
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(data)
# plt.title('Original signal')
# plt.xlabel('time (s)')
# plt.subplot(2,1,2)
# plt.plot(u.T)
# plt.title('Decomposed modes')
# plt.xlabel('time (s)')
# plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
# plt.tight_layout()
# plt.show()

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

for i in range(4):
    data = u[i, :]
    window = signal.windows.gaussian(32, std=0.75)
    f_all = []
    f, t, Zxx = signal.stft(data, fs=160, window=window, nperseg=32, noverlap=26, nfft=38, scaling='psd')
    row = f.shape[0]
    col = t.shape[0]
    result = np.zeros((row*4, col))
    result[i*20:(i+1)*20,:] = np.abs(Zxx)**5
    f_all.append(f)
# hsv = mpl.colormaps.get_cmap('hsv')
# newcmp = ListedColormap(hsv(np.linspace(0.15, 1, 256)))
# fig = plt.pcolormesh(t, f_all, result, cmap=newcmp, vmin=-0, shading='gouraud')
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.colorbar()
# plt.show()