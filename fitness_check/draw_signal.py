import numpy as np
import matplotlib.pyplot as plt
from fitness_check.data_access import PhysioDataset

train_data = PhysioDataset(1, "../data/physionet/eegmmidb/files", train='train', transform=None, preprocess=False)
data = train_data.getdata()  # trials, 12, time 4 5 7
print(data.shape)
data = data[0]
for i in range(data.shape[0]):
    data[i, :] = (data[i, :] - np.min(data[i, :])) / (np.max(data[i, :]) - np.min(data[i, :]))

sts = np.load('sample_1.npy')
data_fake = sts[0]

# Parameters
t = np.linspace(0, 1, 640, endpoint=False)

# Create a new figure of size 10x6 inches
plt.figure(figsize=(10, 6))

# Create the first subplot
plt.subplot(3, 1, 1) # (rows, columns, panel number)
plt.plot(t, data[4], label="ture")
plt.plot(t, data_fake[4], label="fake")
plt.title('C3')
plt.ylabel('Amplitude')
plt.legend()

# Create the second subplot
plt.subplot(3, 1, 2)
plt.plot(t, data[5], label="true")
plt.plot(t, data_fake[5], label="fake")
plt.title('C4')
plt.ylabel('Amplitude')
plt.legend()

# Create the third subplot
plt.subplot(3, 1, 3)
plt.plot(t, data[6], label="true")
plt.plot(t, data_fake[6], label="fake")
plt.title('C1')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

# Automatically adjust subplot spaces
plt.tight_layout()

# Display the figure
plt.show()

