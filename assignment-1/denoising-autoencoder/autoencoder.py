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

def corrupt_input(X):
    noise = torch.FloatTensor(np.random.binomial(1, 0.5, size=X.data.size()))
    return Variable(X.data * noise)


def main():
    N = 5
    d_in = 3
    d_out = 2
    dtype = torch.FloatTensor

    ae = Autoencoder(n_visible=d_in, n_hidden=d_out, batch_size=N)
    optimizer = torch.optim.SGD(ae.parameters(), lr=0.01)
    epochs = 10

    X = Variable(torch.randn(N, d_in).type(dtype), requires_grad=False)

    # Training
    for e in range(epochs):
        # corrupt the input
        tilde_x = corrupt_input(X)
        optimizer.zero_grad()
        Z = ae.forward(tilde_x)
        loss = - torch.sum(X * torch.log(Z) + (1.0 - X) * torch.log(1.0 - Z), 1)  # check if you need to give axis
        cost = torch.mean(loss)
        cost.backward()
        optimizer.step()
        print(cost.data)


if __name__ == "__main__":
    main()
