from __future__ import print_function

import sys
sys.path.append("/home/ak6179/jaa-dl/assignment-1/")

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import pickle
from torch.optim import Adam

import argparse


class Encoder(torch.nn.Module):
    def __init__(self, d_in, d_out, activation_type, train_batch_norm, bias, add_noise):
        super(Encoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.activation_type = activation_type
        self.train_batch_norm = train_batch_norm
        self.add_noise = add_noise

        # Encoder
        # Encoder only uses W matrix, no bias
        # TODO: Add initialization for bias if needed
        self.linear = torch.nn.Linear(d_in, d_out, bias=bias)
        self.linear.weight.data = torch.randn(self.linear.weight.data.size()) / np.sqrt(d_in)

        # Batch Normalization
        # For Relu Beta, Gamma of batch-norm are redundant, hence not trained
        # For Softmax Beta, Gamm are trained
        self.batch_norm_no_noise = torch.nn.BatchNorm1d(d_out, affine=False)
        self.batch_norm = torch.nn.BatchNorm1d(d_out, affine=train_batch_norm)

        # Activation
        if activation_type == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_type == 'log_softmax':
            self.activation = torch.nn.LogSoftmax()
        elif activation_type == 'softmax':
            self.activation = torch.nn.Softmax()

    def forward_clean(self, h):
        t = self.linear(h)
        z = self.batch_norm(t)
        h = self.activation(z)
        return h

    def forward_noise(self, tilde_h):
        # The below z_pre will be used in the decoder cost
        z_pre = self.linear(tilde_h)
        z_pre_norm = self.batch_norm_no_noise(z_pre)
        # Add noise
        z_noise = z_pre_norm + Variable(torch.randn(z_pre_norm.size()))
        z = self.batch_norm(z_noise)
        h = self.activation(z)
        return h

    def forward(self, h):
        if self.add_noise:
            return self.forward_noise(h)
        else:
            return self.forward_clean(h)


class StackedEncoders(torch.nn.Module):
    def __init__(self, d_in, d_encoders, activation_types, train_batch_norms, biases, add_noise):
        super(StackedEncoders, self).__init__()
        self.encoders_ref = []
        self.encoders = torch.nn.Sequential()
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
            encoder = Encoder(d_input, d_output, activation, train_batch_norm, bias, add_noise=add_noise)
            self.encoders_ref.append(encoder_ref)
            self.encoders.add_module(encoder_ref, encoder)


    def forward(self, x):
        h = x
        for e_ref in enumerate(self.encoders_ref):
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward(h)
        return h


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description="Parser for Ladder network")
    parser.add_argument('--batch', type=int)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epochs

    print("=====================")
    print("BATCH SIZE:", batch_size)
    print("EPOCHS:", epochs)
    print("=====================\n")

    print("======  Loading Data ======")
    with open("../../data/train_labeled.p") as f:
        train_dataset = pickle.load(f)
    with open("../../data/validation.p") as f:
        valid_dataset = pickle.load(f)

    loader_kwargs = {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    encoder_sizes = [1000, 500, 250, 250, 250, 10]
    encoder_activations = ["relu", "relu", "relu", "relu", "relu", "log_softmax"]
    encoder_train_batch_norms = [False, False, False, False, False, True]
    # TODO: Verify if all the encoders don't have any bias
    encoder_bias = [False, False, False, False, False, False]
    add_noise = False

    se = StackedEncoders(28 * 28, encoder_sizes, encoder_activations,
                         encoder_train_batch_norms, encoder_bias, add_noise)

    optimizer = Adam(se.parameters(), lr=0.002)
    loss = torch.nn.NLLLoss()

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
