# Code taken from:
# https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/5_convolutional_net.py

from __future__ import print_function

import numpy as np
import os
from os import path
import gzip
import urllib
from argparse import ArgumentParser

import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import pickle

DATASET_DIR = 'data/'


# Separately create two sequential here since PyTorch doesn't have nn.View()
class ConvNet(torch.nn.Module):
    def __init__(self, output_dim, dropout_1=0.25, dropout_2=0.25, dropout_3=0.25,
                 dropout_4=0.25, dropout_5=0.5, dropout_6=0.5):
        super(ConvNet, self).__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module('conv_1', torch.nn.Conv2d(1, 8, kernel_size=3))
        self.conv.add_module('batch_1', torch.nn.BatchNorm2d(8, affine=True))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("dropout_1", torch.nn.Dropout2d(p=dropout_1))

        self.conv.add_module('conv_2', torch.nn.Conv2d(8, 16, kernel_size=3))
        self.conv.add_module('batch_2', torch.nn.BatchNorm2d(16, affine=True))
        self.conv.add_module("relu_2", torch.nn.ReLU())
        self.conv.add_module("dropout_2", torch.nn.Dropout2d(p=dropout_2))
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))

        self.conv.add_module('conv_3', torch.nn.Conv2d(16, 32, kernel_size=3))
        self.conv.add_module('batch_3', torch.nn.BatchNorm2d(32, affine=True))
        self.conv.add_module("relu_3", torch.nn.ReLU())
        self.conv.add_module("dropout_3", torch.nn.Dropout2d(p=dropout_3))
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))

        self.conv.add_module('conv_4', torch.nn.Conv2d(32, 64, kernel_size=2))
        self.conv.add_module('batch_4', torch.nn.BatchNorm2d(64, affine=True))
        self.conv.add_module("relu_4", torch.nn.ReLU())
        self.conv.add_module("dropout_4", torch.nn.Dropout2d(p=dropout_4))

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(1024, 256))
        self.fc.add_module('batch_5', torch.nn.BatchNorm1d(256, affine=True))
        self.fc.add_module("relu_5", torch.nn.ReLU())
        self.fc.add_module("dropout_5", torch.nn.Dropout(p=dropout_5))

        self.fc.add_module("fc2", torch.nn.Linear(256, 64))
        self.fc.add_module('batch_6', torch.nn.BatchNorm1d(64, affine=True))
        self.fc.add_module("relu_6", torch.nn.ReLU())
        self.fc.add_module("dropout_6", torch.nn.Dropout(p=dropout_6))

        self.fc.add_module("fc3", torch.nn.Linear(64, output_dim))
        self.fc.add_module("softmax", torch.nn.Softmax())

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 1024)
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

def run(train_loader, valid_data, valid_label, n_examples, epochs, d1, d2, d3, d4, d5, d6, minibatch, save_label):
    torch.manual_seed(42)

    n_classes = 10
    model = model = ConvNet(output_dim=n_classes,
                            dropout_1=d1,
                            dropout_2=d2,
                            dropout_3=d3,
                            dropout_4=d4,
                            dropout_5=d5,
                            dropout_6=d6)
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    batch_size = minibatch
    for i in range(epochs):
        model.train()
        cost = 0.
        for ind, (data, label) in enumerate(train_loader):
            cost += train(model, loss, optimizer, data, label)

        model.eval()
        predY = predict(model, valid_data)

        validation_accuracy = 100. * np.mean(predY == valid_label.numpy())
        print("Epoch %d, cost = %f, val_acc = %.2f%%"
              % (i + 1, 
                cost / (n_examples/batch_size),
                validation_accuracy))

    with open("final_model_save/convheavy_{}.model".format(save_label), "w") as f:
        torch.save(model, f)



def main():
    torch.manual_seed(42)

    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', default=1000, help='Number of Epochs To Run')
    parser.add_argument('-d1', '--dropout_1', default=0.25)
    parser.add_argument('-d2', '--dropout_2', default=0.25)
    parser.add_argument('-d3', '--dropout_3', default=0.25)
    parser.add_argument('-d4', '--dropout_4', default=0.25)
    parser.add_argument('-d5', '--dropout_5', default=0.5)
    parser.add_argument('-d6', '--dropout_6', default=0.5)
    parser.add_argument('-b', '--minibatch', default=16)
    parser.add_argument('-l', '--lr', default=0.0001)
    # parser.add_argument()
    args = vars(parser.parse_args())

    use_cuda = False
    momentum_par = 0.5
    lr = 0.01
    log_interval = 1

    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # trainset_labeled = pickle.load(open("data/train_labeled.p", "rb"))

    print('Loading Training Data')
    train_data = pickle.load(open('data/generated_train_data_norm.p', 'rb'))
    train_data = torch.from_numpy(train_data).float()  # .resize_(27000,1,28,28)
    train_label = pickle.load(open('data/generated_train_labels.p', 'rb'))
    train_label = torch.from_numpy(train_label).long()

    print('Loading Validation Data')
    valid_data = pickle.load(open('data/generated_valid_data_norm.p', 'rb'))
    valid_data = torch.from_numpy(valid_data).float().resize_(len(valid_data), 1, 28, 28)

    valid_label = pickle.load(open('data/generated_valid_labels.p', 'rb'))
    valid_label = torch.from_numpy(valid_label).long()

    n_examples = len(train_data)
    n_classes = 10
    model = ConvNet(output_dim=n_classes,
                    dropout_1=float(args['dropout_1']),
                    dropout_2=float(args['dropout_2']),
                    dropout_3=float(args['dropout_3']),
                    dropout_4=float(args['dropout_4']),
                    dropout_5=float(args['dropout_5']),
                    dropout_6=float(args['dropout_6']))
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.Adagrad(model.parameters())
    # optimizer = optim.Adamax(model.parameters())
    batch_size = int(args['minibatch'])

    print('Creating Data Loaders')
    train_loader = DataLoader(TensorDataset(train_data, train_label),
                              batch_size=batch_size,
                              shuffle=True)
    # valid_loader = DataLoader(TensorDataset(valid_data, valid_label))

    epochs = int(args['epochs'])

    print('Training Fun Time!!!')

    best_validation_accuracy = 99.0

    for i in range(epochs):
        # Training Mode
        model.train()
        cost = 0.
        for ind, (data, label) in enumerate(train_loader):
            cost += train(model, loss, optimizer, data, label)

        # Prediction Mode
        model.eval()
        predY = predict(model, valid_data)
        pred_train_y = predict(model, train_data)

        validation_accuracy = 100. * np.mean(predY == valid_label.numpy())

        dropout_infor = str(args['dropout_1']).split('.')[1] + \
                        str(args['dropout_2']).split('.')[1] + \
                        str(args['dropout_3']).split('.')[1] + \
                        str(args['dropout_4']).split('.')[1] + \
                        str(args['dropout_5']).split('.')[1] + \
                        str(args['dropout_6']).split('.')[1]

        model_infor = 'd{}b{}e{}acc{}'.format(dropout_infor,
                                              str(batch_size),
                                              str(i),
                                              str(validation_accuracy).replace('.', ''))

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            with open("saved_models/best_heavy_dropout_{}.model".format(model_infor), "w") as file_pointer:
                torch.save(model, file_pointer)

        print("Epoch %d, cost = %f, train_acc = %.2f%% val_acc = %.2f%%"
              % (i + 1,
                 cost / (n_examples / batch_size),
                 100. * np.mean(pred_train_y == train_label.numpy()),
                 validation_accuracy))

        # with open("latest_cnn_{}.model".format(model_infor), "w") as file_pointer:
        #    torch.save(model, file_pointer)


if __name__ == "__main__":
    main()


