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
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms


use_cuda = False
momentum_par = 0.5
lr = 0.01
log_interval = 10
epochs = 100

loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

torch.manual_seed(42)

print ('Loading trainset')
trainset_labeled = pickle.load(open("data/train_labeled.p", "rb"))
print ('Loading validset')
validset = pickle.load(open("data/validation.p", "rb"))

def zca_whitening(train_data, epsilon=1, valid_data=None):
    #Epsilon is whitening constant, and prevents division by zero
    
    data = train_data.numpy()
    orig_shape = data.shape
    data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
    
    # Correlation matrix requires zero-mean
    mu = np.mean(data, axis=0)
    data=(data-mu).T

    sigma = np.dot(data, data.T)/data.shape[1] #Correlation matrix
    U,S,_ = np.linalg.svd(sigma) #Singular Value Decomposition
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S + epsilon))), U.T)  #ZCA Whitening matrix
    
    whitened_train_data = np.dot(ZCAMatrix, data).T.reshape(orig_shape)
    
    if type(valid_data)==type(None):
        whitened_valid_data=None
    else:
        valid_data=valid_data.numpy()
        orig_valid_shape = valid_data.shape
        valid_data = valid_data.reshape(valid_data.shape[0],valid_data.shape[1]*valid_data.shape[2])
        valid_data=(valid_data-mu).T
        
        whitened_valid_data = np.dot(ZCAMatrix, valid_data).T.reshape(orig_valid_shape)
    
    return ZCAMatrix, whitened_train_data, whitened_valid_data   

print('Running ZCA')
_, white_train_data, white_valid_data = zca_whitening(trainset_labeled.train_data, epsilon=1e9, valid_data = validset.test_data)

white_train_data = torch.from_numpy(white_train_data).float()
white_valid_data = torch.from_numpy(white_valid_data).float()

print ('Building set')
trainset_labeled.train_data = white_train_data
validset.test_data = white_valid_data

train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64, shuffle=True, **loader_kwargs)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)

# network given in assignment
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


model = Net()

if use_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum_par)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch, valid_loader)
