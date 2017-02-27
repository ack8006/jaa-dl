from __future__ import print_function

import sys
sys.path.append("/Users/abhishekkadian/Documents/Github/jaa-dl/assignment-1")
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

        self.a1 = Parameter(0. * torch.ones(1, d_in))
        self.a2 = Parameter(1. * torch.ones(1, d_in))
        self.a3 = Parameter(0. * torch.ones(1, d_in))
        self.a4 = Parameter(0. * torch.ones(1, d_in))
        self.a5 = Parameter(0. * torch.ones(1, d_in))

        self.a6 = Parameter(0. * torch.ones(1, d_in))
        self.a7 = Parameter(1. * torch.ones(1, d_in))
        self.a8 = Parameter(0. * torch.ones(1, d_in))
        self.a9 = Parameter(0. * torch.ones(1, d_in))
        self.a10 = Parameter(0. * torch.ones(1, d_in))

        self.V = torch.nn.Linear(d_in, d_out, bias=False)
        self.V.weight.data = torch.randn(self.V.weight.data.size()) / np.sqrt(d_in)
        # batch-normalization for u
        self.bn_normalize = torch.nn.BatchNorm1d(d_out, affine=False)

        # buffer for hat_z_l to be used for cost calculation
        self.buffer_hat_z_l = None

    def g(self, tilde_z_l, u_l):
        ones = Parameter(torch.ones(tilde_z_l.size()[0], 1))

        b_a1 = ones.mm(self.a1)
        b_a2 = ones.mm(self.a2)
        b_a3 = ones.mm(self.a3)
        b_a4 = ones.mm(self.a4)
        b_a5 = ones.mm(self.a5)

        b_a6 = ones.mm(self.a6)
        b_a7 = ones.mm(self.a7)
        b_a8 = ones.mm(self.a8)
        b_a9 = ones.mm(self.a9)
        b_a10 = ones.mm(self.a10)

        mu_l = torch.mul(b_a1, torch.sigmoid(torch.mul(b_a2, u_l) + b_a3)) + \
               torch.mul(b_a4, u_l) + \
               b_a5

        v_l = torch.mul(b_a6, torch.sigmoid(torch.mul(b_a7, u_l) + b_a8)) + \
              torch.mul(b_a9, u_l) + \
              b_a10

        hat_z_l = torch.mul(tilde_z_l - mu_l, v_l) + mu_l

        return hat_z_l

    def forward(self, tilde_z_l, u_l):
        # hat_z_l will be used for calculating decoder costs
        hat_z_l = self.g(tilde_z_l, u_l)
        # store hat_z_l in buffer for cost calculation
        self.buffer_hat_z_l = hat_z_l
        t = self.V.forward(hat_z_l)
        u_l_below = self.bn_normalize(t)
        return u_l_below


class StackedDecoders(torch.nn.Module):
    def __init__(self, d_in, d_decoders):
        super(StackedDecoders, self).__init__()
        self.decoders_ref = []
        self.decoders = torch.nn.Sequential()
        n_decoders = len(d_decoders)
        for i in range(n_decoders):
            if i == 0:
                d_input = d_in
            else:
                d_input = d_decoders[i - 1]
            d_output = d_decoders[i]
            decoder_ref = "decoder_" + str(i)
            decoder = Decoder(d_input, d_output)
            self.decoders_ref.append(decoder_ref)
            self.decoders.add_module(decoder_ref, decoder)

    def forward(self, tilde_z_layers, u_top):
        # Note that tilde_z_layers should be in reversed order of encoders
        hat_z = []
        u = u_top
        for i in range(len(self.decoders_ref)):
            d_ref = self.decoders_ref[i]
            decoder = getattr(self.decoders, d_ref)
            tilde_z = tilde_z_layers[i]
            u = decoder.forward(tilde_z, u)
            hat_z.append(decoder.buffer_hat_z_l)
        return hat_z


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

        # buffer for z_pre which will be used in decoder cost
        self.buffer_z_pre = None
        # buffer for tilde_z which will be used by decoder for reconstruction
        self.buffer_tilde_z = None

    def bn_normalize(self, x):
        ones = Variable(torch.ones(x.size()[0], 1))
        mean = torch.mean(x, 0)
        std = torch.std(x, 0)
        x_normalized = torch.div(x - ones.mm(mean), ones.mm(std + 1e-5))
        torch.std()

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
        # store z_pre in buffer
        # TODO: Check whether you have to detach this or not.
        self.buffer_z_pre = z_pre.detach().clone()
        z_pre_norm = self.bn_normalize(z_pre)
        # Add noise
        noise = np.random.normal(loc=0.0, scale=self.noise_level, size=z_pre_norm.size())
        noise = Variable(torch.FloatTensor(noise))
        # tilde_z will be used by decoder for reconstruction
        tilde_z = z_pre_norm + noise
        # store tilde_z in buffer
        self.buffer_tilde_z = tilde_z
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

    def get_encoders_tilde_z(self, reverse=True):
        tilde_z_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            tilde_z = encoder.buffer_tilde_z
            tilde_z_layers.append(tilde_z)
        if reverse:
            tilde_z_layers.reverse()
        return tilde_z_layers


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

    encoder_in = 28 * 28
    decoder_in = 10
    encoder_sizes = [1000, 500, 250, 250, 250, decoder_in]
    decoder_sizes = [250, 250, 250, 500, 1000, encoder_in]

    encoder_activations = ["relu", "relu", "relu", "relu", "relu", "softmax"]
    # TODO: Verify whether you need affine for relu.
    encoder_train_bn_scaling = [False, False, False, False, False, True]
    encoder_bias = [False, False, False, False, False, False]

    se = StackedEncoders(encoder_in, encoder_sizes, encoder_activations, encoder_train_bn_scaling,
                         encoder_bias, add_noise, noise_std)

    de = StackedDecoders(decoder_in, decoder_sizes)

    optimizer = Adam(se.parameters(), lr=0.002)
    loss_labelled = torch.nn.CrossEntropyLoss()

    print("")
    print("=======NETWORK=======")
    print(se)
    print("=====================")

    print("")
    print("=====================")
    print("TRAINING")

    # TODO: Add annealing of learning rate after 100 epochs

    for e in range(epochs):
        agg_cost = 0.
        num_batches = 0

        # TODO: Check if model.train() and model.eval() has impact over all the submodules including all the bn parameters.
        # Training
        se.train()
        de.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # pass through encoders
            data = data[:,0,:,:].numpy()
            data = data.reshape(data.shape[0], 28 * 28)
            data = torch.FloatTensor(data)
            # TODO: Change from LongTensor to IntTensor. AUtograd has a bug with LongTensor.
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = se.forward(data)

            tilde_z_layers = se.get_encoders_tilde_z(reverse=True)

            # pass through decoders
            hat_z_layers = de.forward(tilde_z_layers, output)

            cost = loss_labelled.forward(output, target)
            cost.backward()
            agg_cost += cost.data[0]
            optimizer.step()
            num_batches += 1

        # Evaluation
        se.eval()
        de.eval()
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
