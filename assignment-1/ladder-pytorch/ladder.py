import pickle as pkl
from argparse import ArgumentParser

import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import TensorDataset, DataLoader



class MLP(torch.nn.Module):
    def __init__(self, output_dim=10, dropout=0.5):
        super(MLP, self).__init__()
#Need Whitenoise
        self.mlp = torch.nn.Sequential()
        self.mlp.add_module('l1', torch.nn.Linear(784, 1000))
        self.mlp.add_module('bn1', torch.nn.BatchNorm1d(1000))
        self.mlp.add_module("relu_1", torch.nn.ReLU())
        self.mlp.add_module('l2', torch.nn.Linear(1000, 500))
        self.mlp.add_module('bn2', torch.nn.BatchNorm1d(500))
        self.mlp.add_module("relu_2", torch.nn.ReLU())
        self.mlp.add_module('l3', torch.nn.Linear(500, 250))
        self.mlp.add_module('bn3', torch.nn.BatchNorm1d(250))
        self.mlp.add_module("relu_3", torch.nn.ReLU())
        self.mlp.add_module('l4', torch.nn.Linear(250, 250))
        self.mlp.add_module('bn4', torch.nn.BatchNorm1d(250))
        self.mlp.add_module("relu_4", torch.nn.ReLU())
        self.mlp.add_module('l5', torch.nn.Linear(250, 250))
        self.mlp.add_module('bn5', torch.nn.BatchNorm1d(250))
        self.mlp.add_module("relu_5", torch.nn.ReLU())
        self.mlp.add_module('l6', torch.nn.Linear(250, 10))
        self.mlp.add_module('bn6', torch.nn.BatchNorm1d(10))
        self.mlp.add_module("relu_6", torch.nn.ReLU())
        self.mlp.add_module('softmax', torch.nn.Softmax())

    def forward(self, x):
        return self.mlp.forward(x)


def main():
    BATCH_SIZE = 32

    torch.manual_seed(42)

    train_labeled = pickle.load(open("data/train_labeled.p", "rb"))
    train_unlabeled = pickle.load(open("data/train_unlabeled.p", "rb"))
    valid = pickle.load(open("data/validation.p", "rb"))

    train_lab_loader = torch.utils.data.DataLoader(trainset_labeled,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    train_unlab_loader = torch.utils.data.DataLoader(train_unlabeled,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)


if __name__ == '__main__':
    main()
