# matches to vsmscnnv2

import matplotlib.pyplot as plt
from numpy import fabs
from numpy.core.numeric import outer
from util.data_loader import BCI2aDataset, PhysioDataset
from util.transforms import filterBank, gammaFilter, MSD, Energy_Wavelet, Normal
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, det_curve
from model.vsmscnnv2 import VSMSCNN
from util.earlystopper import EarlyStopper

import os
import torch
import numpy as np
import sys
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# folder to load config file
CONFIG_PATH = "./config/"


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
        print(config)

    return config


def train(dataloader, model, loss_fn, optimizer, size):
    size = size
    model.train()
    correct = 0
    train_losses = []
    for batch, data in enumerate(dataloader):
        X_1, y_1 = data[0]
        X_1, y_1 = X_1.to(device, dtype=torch.float), y_1.to(device)
        X_2, y_2 = data[1]
        X_2, y_2 = X_2.to(device, dtype=torch.float), y_2.to(device)
        # Compute prediction error
        pred = model(X_1, X_2)
        loss = loss_fn(pred, y_1)
        correct += (pred.argmax(1) == y_1).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X_1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    print(f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%\n")

    return train_losses, correct

def valid(dataloader, model, loss_fn, size):
    size = size
    model.eval()
    test_loss, correct = 0, 0
    valid_losses = []
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            X_1, y_1 = data[0]
            X_1, y_1 = X_1.to(device, dtype=torch.float), y_1.to(device)
            X_2, y_2 = data[1]
            X_2, y_2 = X_2.to(device, dtype=torch.float), y_2.to(device)
            pred = model(X_1, X_2)
            valid_loss = loss_fn(pred, y_1)
            valid_losses.append(valid_loss.item())
            correct += (pred.argmax(1) == y_1).type(torch.float).sum().item()
    correct /= size
    print(f"Valid Error: \n Accuracy: {(100 * correct):>0.1f}%")
    epoch_acc = 100.0 * correct
    return epoch_acc, valid_losses

def test(dataloader, model):
    model.eval()
    scores = []
    all_y = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, dtype=torch.float), y.to(device)
            pred = model(X)
            scores.extend(10 ** pred.cpu().numpy()[:, 1:])
            all_y.extend(y.cpu().numpy())
    for i, score in enumerate(scores):
        scores[i] = score[0]
    fpr, fnr, thresholds = det_curve(all_y, scores, pos_label=1)
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print("EER: {}%".format(EER * 100))

if __name__ == '__main__':
    config = load_config(sys.argv[1])

    if config["transform"]["name"] == "nBand":
        filterTransform = filterBank([[1,4],[4,8],[8,13],[13,30]], 160)

    elif config["transform"]["name"] == "filter_msd":
        filterTransform = transforms.Compose([gammaFilter(), MSD()])

    elif config["transform"]["name"] == "gammaFilter":
        filterTransform = gammaFilter()

    elif config["transform"]["name"] == "Energy_Wavelet":
        filterTransform = Energy_Wavelet()

    elif config["transform"]["name"] == "AlphaBeta":
        filterTransform = gammaFilter(band=[8, 30])

    elif config["transform"]["name"] == "Alpha":
        filterTransform = gammaFilter(band=[8, 12])

    elif config["transform"]["name"] == "Normal":
        filterTransform = Normal()

    elif config["transform"]["name"] == "None":
        filterTransform = None

    if config["dataset"]["name"] == "PhysioDataset":
        train_data_1 = PhysioDataset(subject=config["train"]["subject"], path=config["dataset"]["location"],
                                   train="train", transform=filterTransform, channels=config["channel"]["select"],
                                   use_channel_no=config["channel"]["number"])
        train_data_2 = PhysioDataset(subject=config["train"]["subject"], path=config["dataset"]["location"],
                                     train="train", transform=None,
                                     channels=config["channel"]["select"],
                                     use_channel_no=config["channel"]["number"])
        if config["train"]["batch_size"] == "all":
            batch_size = len(train_data_1)
        else:
            batch_size = config["train"]["batch_size"]
        train_dataloader_1 = DataLoader(train_data_1, batch_size=batch_size, shuffle=True, drop_last=True)
        train_dataloader_2 = DataLoader(train_data_2, batch_size=batch_size, shuffle=True, drop_last=True)
        size_train = len(train_dataloader_1.dataset)

        # channels = train_data_1.channels()  # get the channel number

        valid_data_1 = PhysioDataset(subject=config["train"]["subject"], path=config["dataset"]["location"],
                                    train="valid", transform=filterTransform, channels=config["channel"]["select"])
        valid_data_2 = PhysioDataset(subject=config["train"]["subject"], path=config["dataset"]["location"],
                                     train="valid", transform=None, channels=config["channel"]["select"])
        if config["valid"]["batch_size"] == "all":
            batch_size = len(valid_data_1)
        else:
            batch_size = config["valid"]["batch_size"]
        valid_dataloader_1 = DataLoader(valid_data_1, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader_2 = DataLoader(valid_data_2, batch_size=batch_size, shuffle=True, drop_last=True)
        size_valid = len(valid_dataloader_1.dataset)

    elif config["dataset"]["name"] == "BCI2aDataset":
        train_data_1 = BCI2aDataset(subject=config["train"]["subject"], path=config["dataset"]["location"], train="train",
                                  transform=filterTransform, select_channel=config["channel"]["select"])
        train_data_2 = BCI2aDataset(subject=config["train"]["subject"], path=config["dataset"]["location"],
                                    train="train", transform=None, select_channel=config["channel"]["select"])
        if config["train"]["batch_size"] == "all":
            batch_size = len(train_data_1)
        else:
            batch_size = config["train"]["batch_size"]
        train_dataloader_1 = DataLoader(train_data_1, batch_size=batch_size, shuffle=True, drop_last=True)
        train_dataloader_2 = DataLoader(train_data_2, batch_size=batch_size, shuffle=True, drop_last=True)
        # channels = train_data.channels()

    if config["model"]["name"] in ["FBCNet", "MICNN", "CNN_LSTM", "CP_MIXEDNET", "EEGNet", "MIXED_FBCNet", "VSMSCNN"]:
        if config["model"]["name"] == "VSMSCNN":
            model = VSMSCNN(num_channels=config["channel"]["number"]).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["initial_lr"],
                                     weight_decay=config["optimizer"]["weight_decay"])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

    print('start training')
    patience = 20
    avg_train_losses = []
    avg_valid_losses = []
    accuracy_list = []
    epochs = config["train"]["epochs"]
    epoch_len = len(str(epochs))
    early_stopper = EarlyStopper(patience=10, min_delta=0.05)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_losses, accuracy = train(zip(train_dataloader_1, train_dataloader_2), model, loss_fn, optimizer, size=size_train)
        train_loss = np.average(train_losses)
        exp_lr_scheduler.step()
        epoch_acc, valid_losses = valid(zip(valid_dataloader_1, valid_dataloader_2), model, loss_fn, size=size_valid)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        accuracy_list.append(accuracy)
        avg_valid_losses.append(valid_loss)
        print_msg = (f'[{t:>0{epoch_len}}/{epochs:>0{epoch_len}}] ' + f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' + f'valid_acc: {epoch_acc:.5f}')
        print(print_msg)
                #     torch.save(model.state_dict(), "../trained/"+str(config["train"]["subject"])+"_"+str(intra_acc)+"_"+str(inter_acc)+"_20"+".pth")
        # if early_stopper.early_stop(valid_loss):
        #     break


    fig, ax = plt.subplots()
    x = range(1, len(accuracy_list) + 1)
    y1 = accuracy_list
    y2 = avg_train_losses
    y3 = avg_valid_losses
    ax.plot(x, y1, label='train accuracy')
    ax.plot(x, y2, label='train loss')
    ax.plot(x, y3, label='valid loss')
    ax.set_xlabel('epochs')
    ax.legend()
    plt.show()
    # plt.savefig(“accuracy_loss.jpg”)


    print("Done!")