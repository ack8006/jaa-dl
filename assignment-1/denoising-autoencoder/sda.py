from  __future__ import print_function

import sys
sys.path.append("/Users/abhishekkadian/Documents/Github/jaa-dl/assignment-1/")

import numpy as np

import torch
from torch.autograd import Variable
from torch.optim import SGD

import autoencoder as ae
import convnet


class SDA(torch.nn.Module):
    """Stacked Denoising Autoencoder

    reference: http://www.deeplearning.net/tutorial/SdA.html,
               http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf
    """
    def __init__(self, d_input, d_hidden_autoencoders, d_out,
                 corruptions, batch_size, pre_lr=0.001, ft_lr=0.1, ft_reg=0.0001):
        super(SDA, self).__init__()
        self.d_input = d_input
        self.d_hidden_autoencoders = list(d_hidden_autoencoders)
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
        self.top_linear1 = torch.nn.Linear(d_hidden_autoencoders[-1], d_out, bias=True)
        self.top_linear1.weight.data.uniform_(-4. * np.sqrt(6. / (d_hidden_autoencoders[-1] + d_out)),
                                              4. * np.sqrt(6. / (d_hidden_autoencoders[-1] + d_out)))
        self.top_linear1.bias.data = torch.zeros(d_out)
        self.top_softmax = torch.nn.Softmax()

    def pretrain(self, x, pt_epochs, verbose=True):
        n = x.data.size()[0]
        num_batches = n / self.batch_size
        t = x
        for i, ae in enumerate(self.autoencoders):
            if i > 0:
                temp = Variable(torch.FloatTensor(n, ae.d_in), requires_grad=False)
                for k in range(num_batches):
                    start, end = k * self.batch_size, (k + 1) * self.batch_size
                    temp.data[start:end] = self.autoencoders[i - 1].encode(t[start:end]).data
                t = temp
            optimizer = SGD(ae.parameters(), lr=self.pre_lr)
            print("Pre-training Autoencoder:", i)
            for ep in range(pt_epochs):
                agg_cost = 0.
                for k in range(num_batches):
                    start, end = k * self.batch_size, (k + 1) * self.batch_size
                    bt = t[start:end]
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
        t = self.top_softmax.forward(t)
        return t

    def finetune(self, train_X, train_y, valid_X, valid_y,
                 valid_actual_size, ft_epochs, verbose=True):
        n = train_X.data.size()[0]
        num_batches = n / self.batch_size
        n_v = valid_X.data.size()[0]
        num_batches_v = n_v / self.batch_size
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
            preds = np.zeros((n_v, self.d_out))
            for k in range(num_batches_v):
                start, end = k * self.batch_size, (k + 1) * self.batch_size
                bX = valid_X[start:end]
                p = self.forward(bX).data.numpy()
                preds[start:end] = p
            correct = 0
            for actual, prediction in zip(valid_y[:valid_actual_size], preds[:valid_actual_size]):
                ind = np.argmax(prediction)
                actual = actual.data.numpy()
                if ind == actual:
                    correct += 1
            if verbose:
                print("Fine-tuning Epoch:", ef, "Cost:", agg_cost.data[0],
                      "Validation Accuracy:", "{0:.4f}".format(correct / float(n_v)))


def main():
    trX, teX, trY, teY = convnet.load_mnist(onehot=False)
    trX = np.array([x.flatten() for x in trX])
    teX = np.array([x.flatten() for x in teX])
    trX = Variable(torch.from_numpy(trX).float())
    teX = Variable(torch.from_numpy(teX).float())
    trY = Variable(torch.from_numpy(trY).long())
    teY = Variable(torch.from_numpy(teY).long())

    batch_size = 64

    actual_size = teX.size()[0]
    padded_size = (actual_size / batch_size + 1) * batch_size
    teX_padded = Variable(torch.FloatTensor(padded_size, teX.size()[1]))
    teY_padded = Variable(torch.LongTensor(padded_size) * 0)
    teX_padded[:actual_size] = teX
    teY_padded[:actual_size] = teY

    sda = SDA(d_input=784,
              d_hidden_autoencoders=[1000, 1000, 1000],
              d_out=10,
              corruptions=[.1, .2, .3],
              batch_size=batch_size)

    sda.pretrain(trX, pt_epochs=15)
    sda.finetune(trX, trY, teX_padded, teY_padded,
                 valid_actual_size=actual_size, ft_epochs=36)


if __name__ == "__main__":
    main()