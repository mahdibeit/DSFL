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
from scipy.special import softmax


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_data(seed):
    class DatasetNonIID(Dataset):
        def __init__(self, dataset, seed):
            seed = int(seed)
            targets = [[0, 1], [0, 1], [2, 3], [2, 3], [4, 5], [4, 5],
                       [6, 7], [6, 7], [8, 9], [8, 9]]
            self.userdataset = [(img, label) for img, label in dataset
                                if label in targets[seed]]
    
        def __len__(self):
            return len(self.userdataset)
    
        def __getitem__(self, item):
    
            image, label = self.userdataset[item]
            return image, label
        
    class DatasetSplit(Dataset):
        def __init__(self, dataset, seed):
            self.dataset = dataset
            length=int(len(dataset)/10)
            self.idxs = list(np.arange(length*seed,length*seed+length))
    
        def __len__(self):
            return len(self.idxs)
    
        def __getitem__(self, item):
            image, label = self.dataset[self.idxs[item]]
            return image, label 
    
    
    transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                    )
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    dataloader = DataLoader(DatasetNonIID(testset, seed),
                            batch_size=32,  # according to your device memory
                            shuffle=False,  # Don't forget to seed your dataloader
                            )
    return dataloader

def findK(comCapacity):
    models = [Net() for _ in range(10)]  # List of models
    
    
    for idx, model in enumerate(models):
        model.load_state_dict(torch.load(f'CIFAR/Model{idx}.pt'))
    
    global_model = Net()
    global_model.load_state_dict(torch.load('CIFARGL/ModelGL.pt'))
    
    results = []
    for idx1, first_model in enumerate(models):
    
        cka = CKA(first_model, global_model,
                  model1_name=f"ClientFirst",   # good idea to provide names to avoid confusion
                  model2_name=f"Global",   
                  # model1_layers=layer_names_resnet18, # List of layers to extract features from
                  # model2_layers=layer_names_resnet34, # extracts all layer features by default
                  device='cuda')
    
        cka.compare(load_data(idx1))  # secondary dataloader is optional
        results.append(cka.export())  # returns a dict that contains model names, layer names

    layer_size = [450, 6, 2400, 16, 48000, 120, 10080, 84, 840, 10]
    
    
    k_list = []
    for user in range(10):
        l = []
        for idx, layer in enumerate([1, 3, 4, 5, 6]):
            l.append((results[user]['CKA'][layer][layer].item()))
        arr = np.array(l)
        arr /= sum(arr)

        k_list.append(arr.tolist())

    
    print(k_list)
    
    return k_list

if __name__ == "__main__":
    findK(728)

