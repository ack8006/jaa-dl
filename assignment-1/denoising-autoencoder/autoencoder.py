from  __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class Autoencoder(torch.nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(Autoencoder, self).__init__()
        self.W = Parameter(torch.Tensor(n_visible, n_hidden))
        self.W.data.uniform_(-4 * np.sqrt(6. / (n_hidden + n_visible)),
                             4 * np.sqrt(6. / (n_hidden + n_visible)))
        self.b = Parameter(torch.zeros(n_hidden))
        self.b_prime = Parameter(torch.zeros(n_visible))

    def forward(self, x):
        # Make this compatible for the mini-batch case when x contains N examples
        u = x.mm(self.W)
        t = torch.add(u, self.b)
        t = torch.sigmoid(t)
        t = t.mm(self.W.transpose(1, 0)) + self.b_prime
        t = torch.sigmoid(t)
        return t

N = 1
d_in = 3
d_out = 2
dtype = torch.FloatTensor

ae = Autoencoder(n_visible=d_in, n_hidden=d_out)
optimizer = torch.optim.SGD(ae.parameters(), lr=0.01)

x = Variable(torch.randn(N, d_in).type(dtype), requires_grad=False)
tilde_x = x.clone()
# tilde_x # corrupt the input

# Training
optimizer.zero_grad()
z = ae.forward(tilde_x)
loss = - torch.sum(x * torch.log(z) + (1.0 - x) * torch.log(1.0 - z)) # check if you need to give axis
cost = torch.mean(loss)
cost.backward()
optimizer.step()


