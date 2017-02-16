from  __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class Autoencoder(torch.nn.Module):
    def __init__(self, n_visible, n_hidden, batch_size):
        super(Autoencoder, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.W = Parameter(torch.Tensor(n_visible, n_hidden), requires_grad=True)
        self.W.data.uniform_(-4 * np.sqrt(6. / (n_hidden + n_visible)),
                             4 * np.sqrt(6. / (n_hidden + n_visible)))
        self.b = Parameter(torch.zeros(1, n_hidden), requires_grad=True)
        self.b_prime = Parameter(torch.zeros(1, n_visible), requires_grad=True)

    def forward(self, X):
        ones = Parameter(torch.ones(self.batch_size, 1))
        t = X.mm(self.W)
        t = t + ones.mm(self.b)
        t = torch.sigmoid(t)
        t = t.mm(self.W.transpose(1, 0)) + ones.mm(self.b_prime)
        t = torch.sigmoid(t)
        return t


def corrupt_input(X, corruption_level=0.5):
    noise = torch.FloatTensor(np.random.binomial(1, corruption_level, size=X.data.size()))
    return Variable(X.data.clone() * noise)


def main():
    N = 1000
    d_in = 784
    d_out = 500
    dtype = torch.FloatTensor
    batch_size = 32

    ae = Autoencoder(n_visible=d_in, n_hidden=d_out, batch_size=batch_size)
    optimizer = torch.optim.SGD(ae.parameters(), lr=0.01)
    epochs = 20

    X = Variable(torch.randn(N, d_in).type(dtype), requires_grad=False)

    # Training
    for e in range(epochs):
        agg_cost = 0.
        num_batches = N / batch_size
        for k in range(num_batches):
            start, end = k * (batch_size), (k + 1) * batch_size
            bX = X[start:end]
            # corrupt the input
            tilde_x = corrupt_input(bX, corruption_level=0.5)
            optimizer.zero_grad()
            Z = ae.forward(tilde_x)
            loss = - torch.sum(bX * torch.log(Z) + (1.0 - bX) * torch.log(1.0 - Z), 1)
            cost = torch.mean(loss)
            cost.backward()
            optimizer.step()
            agg_cost += cost
        agg_cost /= num_batches
        print("epoch:", str(e) + ", cost:", agg_cost.data[0])


if __name__ == "__main__":
    main()
