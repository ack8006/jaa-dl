from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

use_cuda = False
momentum_par = 0.5
lr = 0.01
log_interval = 10
epochs = 30

loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

torch.manual_seed(42)

trainset_labeled = pickle.load(open("../data/train_labeled.p", "rb"))
validset = pickle.load(open("../data/validation.p", "rb"))

train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64, shuffle=True, **loader_kwargs)

Variable
