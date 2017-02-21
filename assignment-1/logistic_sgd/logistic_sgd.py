from __future__ import print_function

import torch
from torch.autograd import Variable
from torch.optim import SGD
import gzip
import os
import pickle
import numpy as np


class LogisticRegression(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(LogisticRegression, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.sequential = torch.nn.Sequential()
        self.sequential.add_module('linear1', torch.nn.Linear(d_in, d_out, bias=True))
        self.sequential.linear1.weight.data = torch.zeros(self.sequential.linear1.weight.size())
        self.sequential.linear1.bias.data = torch.zeros(self.sequential.linear1.bias.size())
        self.sequential.add_module('softmax1', torch.nn.LogSoftmax())

    def forward(self, x):
        return self.sequential.forward(x)


def load_data(dataset='mnist.pkl.gz'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        # dataset = "/scratch/sla382/abhishek/sda-theano/data/mnist.pkl.gz"
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ @Abhishek: Changed from the theano LR implementation

        Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        return data_xy[0], data_xy[1]

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def train_lr(batch_size=600, learning_rate=0.13, epochs=75):
    #-----------------------------------------------------------------------------
    # Prepare the data
    rval = load_data()
    train_X, train_y = rval[0]
    valid_X, valid_y = rval[1]

    trX = np.array([x.flatten() for x in train_X])
    valX = np.array([x.flatten() for x in valid_X])
    trX = Variable(torch.from_numpy(trX).float())
    valX = Variable(torch.from_numpy(valX).float())
    trY = Variable(torch.from_numpy(train_y).long())
    valY = Variable(torch.from_numpy(valid_y).long())

    # Pad the validation set
    actual_size = valX.size()[0]
    padded_size = (actual_size / batch_size + 1) * batch_size
    valX_padded = Variable(torch.FloatTensor(padded_size, valX.size()[1]))
    valY_padded = Variable(torch.LongTensor(padded_size) * 0)
    valX_padded[:actual_size] = valX
    valY_padded[:actual_size] = valY

    valX = valX_padded
    valY = valY_padded
    # -----------------------------------------------------------------------------

    net = LogisticRegression(28 * 28, 10)
    optimizer = SGD(net.parameters(), lr=learning_rate)
    loss = torch.nn.NLLLoss()

    N = trX.data.size()[0]
    num_batches = N // batch_size

    N_valid = valX.data.size()[0]
    num_batches_v = N_valid // batch_size

    for e in range(epochs):
        # Train LR
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            bX = trX[start:end]
            bY = trY[start:end]
            optimizer.zero_grad()
            p = net.forward(bX)
            cost = loss.forward(p, bY)
            cost.backward()
            optimizer.step()
        preds = np.zeros((N_valid, net.d_out))

        # Predict on Validation set
        for k in range(num_batches_v):
            start, end = k * batch_size, (k + 1) * batch_size
            bX = valX[start:end]
            p = net.forward(bX).data.numpy()
            preds[start:end] = p

        correct = 0
        for actual, prediction in zip(valY[:actual_size], preds[:actual_size]):
            ind = np.argmax(prediction)
            actual = actual.data.numpy()
            if ind == actual:
                correct += 1

        print("Epoch:", e, "Validation Error:", "{0:.4f}".format(100 * (1.0 - (correct / float(actual_size)))))


def main():
    train_lr()


if __name__ == "__main__":
    main()
