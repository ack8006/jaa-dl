from  __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class Autoencoder(torch.nn.Module):
    """Denoising Autoencoder
    Encodes the data preferably in a lower dimension.
    During training the input is encoded and then reconstructed using the latent embedding.
    The matrix used for reconstruction is the transpose of matrix used for encoding.

    For building a Denoising Autoencoder noise has to be added to the training data
    before it is passed through the network. 'corrupt' method allows you to
    add noise to the data.

    reference: http://www.deeplearning.net/tutorial/dA.html#da
    """

    # TODO: Adapt autoencoder for DataLoader.

    def __init__(self, n_visible, n_hidden, batch_size):
        super(Autoencoder, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.W = Parameter(torch.FloatTensor(n_visible, n_hidden), requires_grad=True)
        self.W.data.uniform_(-4. * np.sqrt(6. / (n_hidden + n_visible)),
                             4. * np.sqrt(6. / (n_hidden + n_visible)))
        self.b = Parameter(torch.zeros(1, n_hidden), requires_grad=True)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.b_prime = Parameter(torch.zeros(1, n_visible), requires_grad=True)
        self.sigmoid2 = torch.nn.Sigmoid()

    @staticmethod
    def corrupt(X, corruption=0.2):
        noise = torch.FloatTensor(np.random.binomial(1, 1.0 - corruption, size=X.data.size()))
        return Variable(X.data.clone() * noise)

    def encode(self, x):
        ones = Parameter(torch.ones(self.batch_size, 1))
        t = x.mm(self.W)
        t = t + ones.mm(self.b)
        t = self.sigmoid1.forward(t)
        return t

    def decode(self, x):
        ones = Parameter(torch.ones(self.batch_size, 1))
        t = x.mm(self.W.transpose(1, 0)) + ones.mm(self.b_prime)
        t = self.sigmoid2.forward(t)
        return t

    def forward(self, x):
        t = self.encode(x)
        t = self.decode(t)
        return t

    def train_ae(self, train_X, optimizer, epochs, corruption=0.2, verbose=True):
        N = train_X.data.size()[0]
        for e in range(epochs):
            agg_cost = 0.
            num_batches = N / self.batch_size
            for k in range(num_batches):
                start, end = k * self.batch_size, (k + 1) * self.batch_size
                bX = train_X[start:end]
                tilde_x = self.corrupt(bX, corruption=corruption)
                optimizer.zero_grad()
                Z = self.forward(tilde_x)
                loss = -torch.sum(bX * torch.log(Z) + (1.0 - bX) * torch.log(1.0 - Z), 1)
                cost = torch.mean(loss)
                cost.backward()
                optimizer.step()
                agg_cost += cost
            agg_cost /= num_batches
            if verbose:
                print("Epoch:", e, "cost:", agg_cost.data[0])


def main():
    pass


if __name__ == "__main__":
    main()
