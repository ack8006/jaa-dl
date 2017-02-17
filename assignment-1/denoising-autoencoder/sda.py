from  __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import SGD

import autoencoder as ae


class SDA(torch.nn.Module):
    """Stacked Denoising Autoencoder

    reference: http://www.deeplearning.net/tutorial/SdA.html,
               http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf
    """
    def __init__(self, d_input, d_hidden_autoencoders, d_sigmoid_hidden, d_out,
                 corruptions, batch_size, pre_lr=0.001, ft_lr=0.1, ft_reg=0.0001):
        super(SDA, self).__init__()
        self.d_input = d_input
        self.d_hidden_autoencoders = list(d_hidden_autoencoders)
        self.d_sigmoid_hidden = d_sigmoid_hidden
        self.d_out = d_out
        self.corruptions = corruptions
        self.batch_size = batch_size
        self.pre_lr = pre_lr
        self.ft_lr = ft_lr
        self.ft_reg = ft_reg

        # Create the Autoencoders
        self.autoencoders = []
        for i, (d, c) in enumerate(zip(d_hidden_autoencoders, corruptions)):
            if i == 0:
                curr_input = d_input
            else:
                curr_input = d_hidden_autoencoders[i - 1]
            dna = ae.Autoencoder(curr_input, d, batch_size, corruption=c)
            self.autoencoders.append(dna)

        # Create the Logistic Layer
        self.top_linear1 = torch.nn.Linear(d_hidden_autoencoders[-1], d_sigmoid_hidden, bias=True)
        self.top_linear1.weight.data.uniform(-4. * np.sqrt(6. / (d_hidden_autoencoders[-1] + d_sigmoid_hidden)),
                                             4. * np.sqrt(6. / (d_hidden_autoencoders[-1] + d_sigmoid_hidden)))
        self.top_linear1.bias.data = torch.zeros(d_sigmoid_hidden)
        self.top_sigmoid = torch.nn.Sigmoid()
        self.top_linear2 = torch.nn.Linear(d_sigmoid_hidden, d_out, bias=True)
        self.top_linear2.weight.data = Parameter(torch.zeros(d_out, d_sigmoid_hidden))
        self.top_linear2.bias.data = torch.zeros(d_out)
        self.top_softmax = torch.nn.Softmax()

    def pretrain(self, x, pt_epochs, verbose=True):
        n = x.data.size()[0]
        num_batches = n / self.batch_size
        t = x
        for i, ae in enumerate(self.autoencoders):
            if i > 0:
                t = self.autoencoders[i - 1].encode(t)
            optimizer = SGD(ae.parameters(), lr=self.pre_lr)
            print("Pre-training autoencoder:", i)
            for ep in range(pt_epochs):
                agg_cost = 0.
                for k in range(num_batches):
                    start, end = k * self.batch_size, (k + 1) * self.batch_size
                    bt = x[start:end]
                    optimizer.zero_grad()
                    z = ae.forward(bt)
                    loss = -torch.sum(bt * torch.log(z) + (1.0 - bt) * torch.log(1.0 - z), 1)
                    cost = torch.mean(loss)
                    cost.backward()
                    optimizer.step()
                    agg_cost += cost
                agg_cost /= num_batches
                if verbose:
                    print("Pre-training Autoencoder:", i, "Epoch:", ep, "Cost:", agg_cost.data[0])

    def forward(self, x):
        t = x
        # Forward through the Autoencoder
        for ae in self.autoencoders:
            t = ae.encode(t)
        # Forward through the Logistic layer
        t = self.top_linear1.forward(t)
        t = self.top_sigmoid.forward(t)
        t = self.top_linear2.forward(t)
        t = self.top_softmax.forward(t)
        return t

    def finetune(self, train_X, train_y, ft_epochs, verbose=True):
        n = train_X.data.size()[0]
        num_batches = n / self.batch_size
        t = train_X
        optimizer = SGD(self.parameters(), lr=self.ft_lr, weight_decay=self.ft_reg)
        loss = torch.nn.NLLLoss()
        for ef in range(ft_epochs):
            agg_cost = 0
            for k in range(num_batches):
                start, end = k * self.batch_size, (k + 1) * self.batch_size
                bX = train_X[start:end]
                by = train_y[start:end]
                optimizer.zero_grad()
                p = self.forward(bX)
                cost = loss.forward(p, by)
                agg_cost += cost
                cost.backward()
                optimizer.step()
            agg_cost /= num_batches
            if verbose:
                print("Fine-tuning Epoch:", ef, "Cost:", agg_cost.data[0])


def main():
    pass


if __name__ == "__main__":
    main()