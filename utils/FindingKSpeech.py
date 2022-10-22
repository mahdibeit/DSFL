# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:17:39 2022

@author: Mahdi
"""

from torch_cka import CKA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from torchaudio.datasets import SPEECHCOMMANDS
import random

import os


class Net(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu((x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu((x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu((x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu((x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze()


class DatasetSplit(Dataset):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        samples=[i for i in range(len(dataset))]
        length = int(len(dataset)/10)
        myrandom = random.Random(0)  # Fixiing the seed
        myrandom.shuffle(samples)
        self.idxs = samples[length*seed:length*seed+length]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        out= self.dataset[self.idxs[item]]
        return out


class Net2(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu((x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu((x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu((x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu((x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze()


    
class DatasetNonIID(Dataset):
    def __init__(self, dataset, seed):
        seed = int(seed)
        targets = [['right', 'house', 'go', 'seven', 'backward', 'down', 'bed'], ['right', 'house', 'go', 'seven', 'backward', 'down', 'bed'],
                   ['follow', 'marvin', 'nine', 'three', 'eight', 'left', 'cat'], ['follow', 'marvin', 'nine', 'three', 'eight', 'left', 'cat'],
                   ['happy', 'visual', 'zero', 'stop', 'four', 'tree', 'wow'], ['happy', 'visual', 'zero', 'stop', 'four', 'tree', 'wow'],
                   ['off', 'up', 'six', 'two', 'forward', 'learn', 'five'], ['off', 'up', 'six', 'two', 'forward', 'learn', 'five'],
                   ['sheila', 'bird', 'yes', 'dog', 'no', 'on', 'one'], ['sheila', 'bird', 'yes', 'dog', 'no', 'on', 'one']]

        self.userdataset = [(waveform, temp, label) for waveform, temp, label, *_ in dataset
                            if label in targets[seed]]


    def __len__(self):
        return len(self.userdataset)

    def __getitem__(self, item):

        out = self.userdataset[item]
        return out

def load_data(IID, seed):
    """Load SpeachCommands (training and test set)."""
    
    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset: str = None):
            super().__init__("./", download=True)
    
            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
    
            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]
    
    
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

    
    def label_to_index(word):
    # Return the position of the word in labels
        return torch.tensor(labels.index(word))


    def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
        return labels[index]
    
    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)
    
    
    def collate_fn(batch):
    
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number
    
        tensors, targets = [], []
    
        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [label_to_index(label)]
    
        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets
    
    if IID==True:
        testloader = DataLoader(DatasetSplit(test_set, seed), batch_size=32,collate_fn=collate_fn,pin_memory=True)
    else:
        testloader = DataLoader(DatasetNonIID(test_set, seed), batch_size=32, collate_fn=collate_fn, pin_memory=True)

    return testloader



def findK():

    models = [Net() for _ in range(10)]  # List of models
    
    
    for idx, model in enumerate(models):
        model.load_state_dict(torch.load(f'SPEECH/Model{idx}.pt'))
    
    global_model = Net()
    global_model.load_state_dict(torch.load(f'SpeechGL/ModelGL.pt'))

    results = []
    for idx1, first_model in enumerate(models):
        # print(first_model.state_dict())
        cka = CKA(first_model, global_model,
                  model1_name=f"ClientFirst",   # good idea to provide names to avoid confusion
                  model2_name=f"Global",   
                  model1_layers=['conv1', 'conv2', 'conv3', 'conv4', 'fc1'], # List of layers to extract features from
                  model2_layers=['conv1', 'conv2', 'conv3', 'conv4', 'fc1'], # extracts all layer features by default
                  device='cuda')
    
        cka.compare(load_data(False, idx1))  # secondary dataloader is optional
        results.append(cka.export())  # returns a dict that contains model names, layer names
                                     # and the CKA matrix
 

    
    k_list = []
    for user in range(10):
        l = []
        for idx, layer in enumerate([0, 1, 2, 3, 4]):
            l.append((results[user]['CKA'][layer][layer].item()))
        arr = np.array(l)
        arr /= sum(arr)
        k_list.append(arr.tolist())

    
    print(k_list)
    
    return k_list

if __name__ == "__main__":
    findK()

