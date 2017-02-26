from __future__ import print_function

import sys
sys.path.append("/home/ak6179/jaa-dl/assignment-1/")

import numpy as np
import argparse
import pickle

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import Adam


class Decoder(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(Decoder, self).__init__()

        self.a1 = Parameter(0. * torch.ones(d_in))
        self.a2 = Parameter(1. * torch.ones(d_in))
        self.a3 = Parameter(0. * torch.ones(d_in))
        self.a4 = Parameter(0. * torch.ones(d_in))
        self.a5 = Parameter(0. * torch.ones(d_in))

        self.a6 = Parameter(0. * torch.ones(d_in))
        self.a8 = Parameter(0. * torch.ones(d_in))
        self.a7 = Parameter(1. * torch.ones(d_in))
        self.a9 = Parameter(0. * torch.ones(d_in))
        self.a10 = Parameter(0. * torch.ones(d_in))

        self.V = torch.nn.Linear(d_in, d_out, bias=False)
        self.V.weight.data = torch.randn(self.V.weight.data.size()) / np.sqrt(d_in)
        # batch-normalization for u
        self.bn_normalize = torch.nn.BatchNorm1d(d_out, affine=False)

    def g(self, tilde_z_l, u_l):
        raise NotImplementedError

    def forward(self, tilde_z_l, u_l):
        # hat_z_l will be used for calculating decoder costs
        hat_z_l = self.g(tilde_z_l, u_l)
        t = self.V.forward(hat_z_l)
        u_l_below = self.bn_normalize(t)
        return u_l_below


class StackedDecoders(torch.nn.Module):
    raise NotImplementedError


class Encoder(torch.nn.Module):
    def __init__(self, d_in, d_out, activation_type, train_bn_scaling,
                 bias, add_noise, noise_level):
        super(Encoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.activation_type = activation_type
        self.train_bn_scaling = train_bn_scaling
        self.add_noise = add_noise
        self.noise_level = noise_level

        # Encoder
        # Encoder only uses W matrix, no bias
        self.linear = torch.nn.Linear(d_in, d_out, bias=bias)
        self.linear.weight.data = torch.randn(self.linear.weight.data.size()) / np.sqrt(d_in)

        # Batch Normalization
        # For Relu Beta of batch-norm is redundant, hence only Gamma is trained
        # For Softmax Beta, Gamma are trained
        self.bn_normalize = torch.nn.BatchNorm1d(d_out, affine=False)
        # batch-normalization bias
        self.bn_beta = Parameter(torch.FloatTensor(1, d_out))
        self.bn_beta.data.zero_()
        if self.train_bn_scaling:
            # batch-normalization scaling
            self.bn_gamma = Parameter(torch.FloatTensor(1, d_out))
            self.bn_gamma.data = torch.ones(self.bn_gamma.size())

        # Activation
        if activation_type == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_type == 'log_softmax':
            self.activation = torch.nn.LogSoftmax()
        elif activation_type == 'softmax':
            self.activation = torch.nn.Softmax()

    def bn_gamma_beta(self, x):
        ones = Parameter(torch.ones(x.size()[0], 1))
        t = x + ones.mm(self.bn_beta)
        if self.train_bn_scaling:
            t = torch.mul(t, ones.mm(self.bn_gamma))
        return t

    def forward_clean(self, h):
        t = self.linear(h)
        z = self.bn_normalize(t)
        z = self.bn_gamma_beta(z)
        h = self.activation(z)
        return h

    def forward_noise(self, tilde_h):
        # z_pre will be used in the decoder cost
        z_pre = self.linear(tilde_h)
        z_pre_norm = self.bn_normalize(z_pre)
        # Add noise
        noise = np.random.normal(loc=0.0, scale=self.noise_level, size=z_pre_norm.size())
        noise = Variable(torch.FloatTensor(noise))
        # tilde_z will be used by decoder for reconstruction
        tilde_z = z_pre_norm + noise
        z = self.bn_gamma_beta(tilde_z)
        h = self.activation(z)
        return h

    def forward(self, h):
        if self.add_noise:
            return self.forward_noise(h)
        else:
            return self.forward_clean(h)


class StackedEncoders(torch.nn.Module):
    def __init__(self, d_in, d_encoders, activation_types, train_batch_norms,
                 biases, add_noise, noise_std):
        super(StackedEncoders, self).__init__()
        self.encoders_ref = []
        self.encoders = torch.nn.Sequential()
        self.add_noise = add_noise
        self.noise_level = noise_std
        n_encoders = len(d_encoders)
        for i in range(n_encoders):
            if i == 0:
                d_input = d_in
            else:
                d_input = d_encoders[i - 1]
            d_output = d_encoders[i]
            activation = activation_types[i]
            train_batch_norm = train_batch_norms[i]
            bias = biases[i]
            encoder_ref = "encoder_" + str(i)
            encoder = Encoder(d_input, d_output, activation, train_batch_norm,
                              bias, add_noise=add_noise, noise_level=noise_std)
            self.encoders_ref.append(encoder_ref)
            self.encoders.add_module(encoder_ref, encoder)


    def forward(self, x):
        # add noise
        if self.add_noise:
            noise = np.random.normal(loc=0.0, scale=self.noise_level, size=x.size())
            noise = Variable(torch.FloatTensor(noise))
            h = x + noise
        else:
            h = x
        # pass through encoders
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward(h)
        return h


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description="Parser for Ladder network")
    parser.add_argument('--batch', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--noise_std', type=float)
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epochs
    noise_std = args.noise_std
    add_noise = True

    print("=====================")
    print("BATCH SIZE:", batch_size)
    print("EPOCHS:", epochs)
    print("ADD NOISE:", add_noise)
    print("NOISE STD:", noise_std)
    print("=====================\n")

    print("======  Loading Data ======")
    with open("../../data/train_labeled.p") as f:
        train_dataset = pickle.load(f)
    with open("../../data/validation.p") as f:
        valid_dataset = pickle.load(f)
    print("===========================")

    loader_kwargs = {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    encoder_sizes = [1000, 500, 250, 250, 250, 10]
    encoder_activations = ["relu", "relu", "relu", "relu", "relu", "log_softmax"]
    # TODO: Verify whether you need affine for relu.
    encoder_train_bn_scaling = [False, False, False, False, False, True]
    encoder_bias = [False, False, False, False, False, False]

    se = StackedEncoders(28 * 28, encoder_sizes, encoder_activations, encoder_train_bn_scaling,
                         encoder_bias, add_noise, noise_std)

    optimizer = Adam(se.parameters(), lr=0.002)
    loss = torch.nn.NLLLoss()

    print("")
    print("=======NETWORK=======")
    print(se)
    print("=====================")

    print("")
    print("=====================")
    print("TRAINING")

    for e in range(epochs):
        agg_cost = 0.
        num_batches = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data[:,0,:,:].numpy()
            data = data.reshape(data.shape[0], 28 * 28)
            data = torch.FloatTensor(data)
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = se.forward(data)
            cost = loss.forward(output, target)
            cost.backward()
            agg_cost += cost.data[0]
            optimizer.step()
            num_batches += 1
        agg_cost /= num_batches
        correct = 0.
        total = 0.
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data[:, 0, :, :].numpy()
            data = data.reshape(data.shape[0], 28 * 28)
            data = torch.FloatTensor(data)
            data, target = Variable(data), Variable(target)
            output = se.forward(data)
            output = output.data.numpy()
            preds = np.argmax(output, axis=1)
            target = target.data.numpy()
            correct += np.sum(target == preds)
            total += target.shape[0]
        print("Epoch:", e + 1, "Cost:", agg_cost, "Validation Accuracy:", correct / total)

    print("=====================\n")

    print("Done :)")

if __name__ == "__main__":
    main()
