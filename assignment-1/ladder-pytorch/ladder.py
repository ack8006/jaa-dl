import pickle as pkl
from argparse import ArgumentParser

import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Softmax


#*** What is the g function? Ans: torch impl has three possibles 'vanilla', 'vanilla-rand', 'gaussian'
#Is unlabeled data also fed into both encoders? what is the loss function for unlabeled?
#For labeled I'm fairly sure it's the sum of loss functions


class MLP(torch.nn.Module):
    def __init__(self, output_dim=10, dropout=0.5):
        super(MLP, self).__init__()
        #These are encoder layers
        self.encoders = []
        #Layer Sizes
        self.ls = [784, 1000, 500, 250, 250, 250]
        n_layers = len(ls) #6
        for i in xrange(n_layers - 1):
            self.encoders[i] = Sequential()
            self.encoders[i].add_module('l_{}'.format(i), Linear(self.ls[i], self.ls[i+1]))
#*** eps, momentum are default nonzero, torch imp sets both to 0
#*** affine default to true giving learnable params, torch imp sets to false
            #self.encoders[i].add_module('bn_{}'.format(i), BatchNorm1d(self.ls[i+1], ))
            self.encoders[i].add_module('bn_{}'.format(i), BatchNorm1d(self.ls[i+1], 0.0, 0.0, False))
            self.encoders[i].add_module('relu_{}'.format(i), ReLU())

        self.encoders[n_layers] = Sequential()
        self.encoders[n_layers].add_module('l_n', Linear(self.ls[n_layers], output_dim))
#*** eps, momentum are default nonzero, torch imp doesn't set to 0 in last layer
#*** affine default to true giving learnable params, torch imp  leaves it as true
        #self.encoders[n_layers].add_module('bn_n', BatchNorm1d(output_dim))
        self.encoders[n_layers].add_module('bn_n', BatchNorm1d(output_dim, 0.0, 0.0, False))
        self.encoders[n_layers].add_module('relu_{}', ReLU())

#***This may have to be it's own sequential...unsure
        # TODO: Use LogSoftmax because we have to use Negative Log Likelihood
        self.encoders[n_layers].add_module('softmax', Softmax())

#***Need decoder layers here
        self.decoders = [] #Should also be 6

#***Do something unique for last layer, then can loop through 
        self.decoders[n_layers-1] = Sequential()
        #self.decoders[n_layers-1].add_module(
        #self.decoders[n_layers-1].add_module(

        #4,3,2,1
        for i in xrange(n_layers-2, 0, -1):
            self.decoders[i] = Sequential()
            #self.decoders[i].add_module(BatchNorm1d(


    #Generates Gaussian Noise and add to variable x
    #*** paper recommends batch mean and batch stdev, but not sure what that means
    # Data is 0-1, what happens when subtract from 0 pixel?
    # TODO: Add batch normalization. Related to the statement above.
    def noiseify(x, noise_level = 0.1):
        noise = Variable(torch.normal(means= torch.zeros(x.size()),
                                      std = noise_level))
        return torch.add(x, noise)

    def encode(self, x, noisey=False):
        #Multiple encoders for adding noise
        #https://discuss.pytorch.org/t/whitenoise-layer-for-dcgan-tutorial/422/3

        noise_levels = [0.2]*5

        #***Should likely store at each level as these will be needed for g function
        for i, dim in self.ls[1:]:
            x = self.encoders[i].forward(x)
            x = x.view(-1, dim)
            if noisey:
                x = noiseify(x, noise_level[i])

        return self.encoders[len(self.ls)].forward(x)

    def decode(self, x):
        pass

#*** What is the mean of? z?
    #Get z_hat
    #Implementation of Function 1
    def denoising(self, z_til, z_var, n_var, mean):
        v = z_var / (z_var + n_var)
        z_hat = (z_til - mean) * v + mean
        return z_hat


def main():
    BATCH_SIZE = 32

    torch.manual_seed(42)

    train_labeled = pickle.load(open("data/train_labeled.p", "rb"))
    train_unlabeled = pickle.load(open("data/train_unlabeled.p", "rb"))
    valid = pickle.load(open("data/validation.p", "rb"))

    train_lab_loader = torch.utils.data.DataLoader(trainset_labeled,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    #Look at Piazza for handling missing labels
    train_unlab_loader = torch.utils.data.DataLoader(train_unlabeled,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)


if __name__ == '__main__':
    main()
