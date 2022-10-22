# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 20:36:15 2022

@author: Mahdi
"""

from collections import OrderedDict
import warnings
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common.logger import log
from logging import INFO
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import MNIST
import numpy as np
import random
import argparse





warnings.filterwarnings("ignore", category=Warning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


    


# #############################################################################
# 1. PyTorch pipeline: model/train/test/dataloader
# #############################################################################

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(net, trainloader, epochs,seed):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


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


def load_data(seed, IID):
    """Load Mnist-10 (training and test set)."""
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    testset = MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    if IID==True:
        trainloader = DataLoader(DatasetSplit(trainset, seed), batch_size=32, shuffle=True)
        testloader = DataLoader(DatasetSplit(testset, seed), batch_size=32)
        num_examples = {"trainset": len(trainset), "testset": len(testset)}
        return trainloader, testloader, num_examples
    else:
        trainloader = DataLoader(DatasetNonIID(trainset, seed), batch_size=32, shuffle=True)
        testloader = DataLoader(DatasetNonIID(testset, seed), batch_size=32)
        num_examples = {"trainset": len(trainset), "testset": len(testset)}
        return trainloader, testloader, num_examples




# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def main():
    
    iid = False
    
    """Fixing the seed for reproducability"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="set the seed for reporiducabilty")
    args = parser.parse_args()
    if args.seed:
        seed = args.seed
        log(INFO, f"Using seed {seed} for reproducability")
        torch.manual_seed(seed)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    
    """Create model, load data, define Flower client, start Flower client."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (MNIST)
    trainloader, testloader, num_examples = load_data(int(args.seed),IID=iid)
    if iid:
        log(INFO, "Using IID dataset")
    else:
        log(INFO, "Using Non-IID dataset")
    
    # Flower client
    class CifarClient(fl.client.NumPyClient):
        def __init__(self,args):
            super().__init__()
            self.Global_round=0
            self.args=args
            
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader,  1, self.args.seed,)

            """Utilize the seed number as the IID number
            [Seed Number, Parameters]
            """
            return self.get_parameters(), num_examples["trainset"], {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client("localhost:8080", client=CifarClient(args))


if __name__ == "__main__":
    main()