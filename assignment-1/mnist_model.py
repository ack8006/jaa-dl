from __future__ import print_function, division

import pickle

import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from convnet3_deep3 import run as run_c3d3
from convnet_leaky_relu import run as run_leaky
from convnet_heavy_dropout import run as run_heavy



def load_super_74():
    train_data = pickle.load(open('data/super_train_74_data.p', 'rb'))
    train_data = torch.from_numpy(train_data).float()
    train_label = pickle.load(open('data/super_train_74_labels.p', 'rb'))
    train_label = torch.from_numpy(train_label).long()
    return train_data, train_label

def load_gold():
    train_data = pickle.load(open('data/gold_data.p', 'rb'))
    train_data = torch.from_numpy(train_data).float()
    train_label = pickle.load(open('data/gold_labels.p', 'rb'))
    train_label = torch.from_numpy(train_label).long()
    return train_data, train_label

def load_validation_data():
    valid_data = pickle.load(open('data/generated_valid_data_norm.p', 'rb'))
    valid_data = torch.from_numpy(valid_data).float().resize_(len(valid_data),1,28,28)
    valid_label = pickle.load(open('data/generated_valid_labels.p', 'rb'))
    valid_label = torch.from_numpy(valid_label).long()
    return valid_data, valid_label

def main():
    train_data, train_label = load_super_74()
    valid_data, valid_label = load_validation_data()

    #MODEL 1, c3d3, b16, d1 2, d2 5, 
    train_loader = DataLoader(TensorDataset(train_data, train_label),
                                batch_size = 16,
                                shuffle=True)



    run_c3d3(train_loader, valid_data, valid_label, len(train_data), 119, 0.2, 0.5, 16, 'mdl1')
    run_c3d3(train_loader, valid_data, valid_label, len(train_data), 161, 0.3, 0.4, 16, 'mdl2')

    del train_data, train_label, train_loader
    train_data, train_label = load_gold()
    train_loader = DataLoader(TensorDataset(train_data, train_label),
                                batch_size = 16,
                                shuffle=True)

    run_leaky(train_loader, valid_data, valid_label, len(train_data), 610, 0.4, 0.6, 16, 'mdl2')
    run_heavy(train_loader, valid_data, valid_label, len(train_data), 196, 0.25,0.25,0.25,0.25,0.5,0.5,16,'mld1')

    del train_loader
    train_loader = DataLoader(TensorDataset(train_data, train_label),
                                batch_size = 32,
                                shuffle=True)

    run_heavy(train_loader, valid_data, valid_label, len(train_data), 283, 0.25,0.25,0.25,0.25,0.5,0.5,32,'mld2')
    run_leaky(train_loader, valid_data, valid_label, len(train_data), 767, 0.4, 0.5, 32, 'mdl2')

if __name__ == '__main__':
    main()

