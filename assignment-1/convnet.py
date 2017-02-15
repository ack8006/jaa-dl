# Code taken from:
# https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/5_convolutional_net.py

from __future__ import print_function

import numpy as np
import os
from os import path
import gzip
import urllib
import pickle

import torch
from torch.autograd import Variable
from torch import optim

DATASET_DIR = 'data/'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def download_file(url, local_path):
    dir_path = path.dirname(local_path)
    if not path.exists(dir_path):
        print("Creating the directory '%s' ..." % dir_path)
        os.makedirs(dir_path)

    print("Downloading from '%s' ..." % url)
    urllib.URLopener().retrieve(url, local_path)


def download_mnist(local_path):
    url_root = "http://yann.lecun.com/exdb/mnist/"
    for f_name in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
        f_path = os.path.join(local_path, f_name)
        if not path.exists(f_path):
            download_file(url_root + f_name, f_path)


def load_mnist(ntrain=60000, ntest=10000, onehot=True):
    data_dir = os.path.join(DATASET_DIR, 'mnist/')
    if not path.exists(data_dir):
        download_mnist(data_dir)

    with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    with gzip.open(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        trY = loaded[8:].reshape((60000))

    with gzip.open(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    with gzip.open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        teY = loaded[8:].reshape((10000))

    trX /= 255.
    teX /= 255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY


# Separately create two sequential here since PyTorch doesn't have nn.View()
class ConvNet(torch.nn.Module):
    def __init__(self, output_dim):
        super(ConvNet, self).__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(1, 10, kernel_size=5))
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("conv_2", torch.nn.Conv2d(10, 20, kernel_size=5))
        self.conv.add_module("dropout_2", torch.nn.Dropout())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(320, 50))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout())
        self.fc.add_module("fc2", torch.nn.Linear(50, output_dim))
        self.fc.add_module("relu_4", torch.nn.ReLU())
        self.fc.add_module("softmax", torch.nn.Softmax())

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 320)
        return self.fc.forward(x)


def train(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]


def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)


def main():
    torch.manual_seed(42)
    # trX, teX, trY, teY = load_mnist(onehot=False)

    use_cuda = False
    momentum_par = 0.5
    lr = 0.01
    log_interval = 1

    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    trainset_labeled = pickle.load(open("data/train_labeled.p", "rb"))
    validset = pickle.load(open("data/validation.p", "rb"))

    train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64, shuffle=True, **loader_kwargs)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)

    train_data_batches = []
    train_label_batches = []
    train_data = []
    train_label = []

    for batch_idx, (data, target) in enumerate(train_loader):
        train_data_batches.append(data.numpy()[:, 0, :, :])
        train_label_batches.append(target.numpy())

    for x, y in zip(train_label_batches, train_data_batches):
        for a, b in zip(x, y):
            train_label.append(a)
            train_data.append(b)

    valid_data_batches = []
    valid_label_batches = []
    valid_data = []
    valid_label = []

    for batch_idx, (data, target) in enumerate(valid_loader):
        valid_data_batches.append(data.numpy()[:, 0, :, :])
        valid_label_batches.append(target.numpy())

    for x, y in zip(valid_label_batches, valid_data_batches):
        for a, b in zip(x, y):
            valid_label.append(a)
            valid_data.append(b)

    train_data = np.array(train_data)
    train_label = np.array(train_label)

    valid_data = np.array(valid_data)
    valid_label = np.array(valid_label)

    print("Shapes:",train_data.shape, train_label.shape, valid_data.shape, valid_label.shape)

    torch.manual_seed(42)

    trX = train_data
    teX = valid_data
    trY = train_label
    teY = valid_label

    trX = trX.reshape(-1, 1, 28, 28)
    teX = teX.reshape(-1, 1, 28, 28)

    trX = torch.from_numpy(trX).float()
    teX = torch.from_numpy(teX).float()
    trY = torch.from_numpy(trY).long()

    n_examples = len(trX)
    n_classes = 10
    model = ConvNet(output_dim=n_classes)
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    batch_size = 64

    epochs = 1000

    for i in range(epochs):
        cost = 0.
        num_batches = n_examples / batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, trX[start:end], trY[start:end])
        predY = predict(model, teX)
        print("Epoch %d, cost = %f, acc = %.2f%%"
              % (i + 1, cost / num_batches, 100. * np.mean(predY == teY)))


if __name__ == "__main__":
    main()

