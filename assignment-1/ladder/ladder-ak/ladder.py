import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np


class Encoder(torch.nn.Module):
    def __init__(self, d_in, d_out, activation_type, train_batch_norm, bias, batch_size):
        super(Encoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.activation_type = activation_type
        self.train_batch_norm = train_batch_norm
        self.batch_size = batch_size

        # Encoder
        # Encoder only uses W matrix, no bias
        # TODO: Add initialization for bias if needed
        self.linear = torch.nn.Linear(d_in, d_out, bias=bias)
        self.linear.weight.data = torch.randn(d_in, d_out) / np.sqrt(d_in)

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


class StackedEncoders(torch.nn.Module):
    def __init__(self, d_in, d_encoders, activation_types, train_batch_norms, biases, batch_size):
        super(StackedEncoders, self).__init__()
        self.encoders = []
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
            encoder = Encoder(d_input, d_output, activation, train_batch_norm, bias, batch_size)
            self.encoders.append(encoder)

    def forward_clean(self, x):
        h = x
        for encoder in self.encoders:
            h = encoder.forward_clean(h)
        return h

    def forward_noise(self, x):
        h = x + Variable(torch.randn(x.size()))
        for encoder in self.encoders:
            h = encoder.forward_noise(h)
        return h



def main():
    encoder_sizes = [1000, 500, 250, 250, 250, 10]
    encoder_activations = ["relu", "relu", "relu", "relu", "relu", "log_softmax"]
    encoder_train_batch_norms = [False, False, False, False, False, True]
    # TODO: Verify if all the encoders don't have any bias
    encoder_bias = [False, False, False, False, False, False]

    batch_size = 100

    se = StackedEncoders(28 * 28, encoder_sizes, encoder_activations,
                         encoder_train_batch_norms, encoder_bias, batch_size)

    print("Done")


if __name__ == "__main__":
    main()