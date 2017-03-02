from __future__ import print_function

import sys
sys.path.append("/Users/abhishekkadian/Documents/Github/jaa-dl/assignment-1")
sys.path.append("/home/ak6179/jaa-dl/assignment-1/")

import numpy as np
import argparse
import pickle

import torch
from torch.autograd import Variable
from torch.optim import Adam

from encoder import StackedEncoders
from decoder import StackedDecoders


class Ladder(torch.nn.Module):
    def __init__(self, encoder_in, encoder_sizes, decoder_in, decoder_sizes,
                 encoder_activations, encoder_train_bn_scaling, encoder_bias,
                 noise_std):
        super(Ladder, self).__init__()
        self.se = StackedEncoders(encoder_in, encoder_sizes, encoder_activations,
                                  encoder_train_bn_scaling, encoder_bias, noise_std)
        self.de = StackedDecoders(decoder_in, decoder_sizes)

    def forward_encoders_clean(self, data):
        return self.se.forward_clean(data)

    def forward_encoders_noise(self, data):
        return self.se.forward_noise(data)

    def forward_decoders(self, tilde_z_layers, encoder_output):
        return self.de.forward(tilde_z_layers, encoder_output)

    def get_encoders_tilde_z(self, reverse=True):
        return self.se.get_encoders_tilde_z(reverse)

    def get_encoders_z_pre(self, reverse=True):
        return self.se.get_encoders_z_pre(reverse)

    def get_encoders_z(self, reverse=True):
        return self.se.get_encoders_z(reverse)

    def decoder_bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        return self.de.bn_hat_z_layers(hat_z_layers, z_pre_layers)


def main():
    # TODO IMPORTANT: maintain a different batch-normalization layer for the clean pass
    # otherwise it will mess up the running means and variances for the noisy pass
    # which have to be used in the final prediction. Note that although we do a 
    # clean pass to get the reconstruction targets our supervised cost comes from the
    # noisy pass but our prediction on validation and test set comes from the clean pass.
    
    # TODO: Not so sure about the above clean and noisy pass. Test both versions.

    # TODO: Don't batch normalize using z_pre in the first decoder

    # command line arguments
    parser = argparse.ArgumentParser(description="Parser for Ladder network")
    parser.add_argument('--batch', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--noise_std', type=float)
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epochs
    noise_std = args.noise_std

    print("=====================")
    print("BATCH SIZE:", batch_size)
    print("EPOCHS:", epochs)
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

    ladder = Ladder(encoder_in, encoder_sizes, decoder_in, decoder_sizes, encoder_activations,
                    encoder_train_bn_scaling, encoder_bias, noise_std)

    optimizer = Adam(ladder.parameters(), lr=0.002)
    loss_labelled = torch.nn.CrossEntropyLoss()

    print("")
    print("=======NETWORK=======")
    print(ladder)
    print("=====================")

    print("")
    print("=====================")
    print("TRAINING")

    # TODO: Add annealing of learning rate after 100 epochs

    for e in range(epochs):
        agg_cost = 0.
        num_batches = 0

        # Training
        ladder.train()

        # TODO: Add volatile for the input parameters in training and validation

        for batch_idx, (data, target) in enumerate(train_loader):
            # pass through encoders
            data = data[:,0,:,:].numpy()
            data = data.reshape(data.shape[0], 28 * 28)
            data = torch.FloatTensor(data)
            # TODO: Hold off on this, things should work right now because LongTensor is only used for cost.
            # TODO: Change from LongTensor to FloatTensor. Autograd has a bug with LongTensor.
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()

            # do a noisy pass
            output_noise = ladder.forward_encoders_noise(data)
            tilde_z_layers = ladder.get_encoders_tilde_z(reverse=True)

            # do a clean pass
            output_clean = ladder.forward_encoders_clean(data)
            z_pre_layers = ladder.get_encoders_z_pre(reverse=True)
            z_layers = ladder.get_encoders_z(reverse=True)

            # pass through decoders
            hat_z_layers = ladder.forward_decoders(tilde_z_layers, output_noise)

            # batch normalize using mean, var of z_pe
            bn_hat_z_layers = ladder.decoder_bn_hat_z_layers(hat_z_layers, z_pre_layers)

            cost_supervised = loss_labelled.forward(output_noise, target)
            cost_unsupervised = None

            cost_supervised.backward()

            cost = cost_supervised
            agg_cost += cost.data[0]
            optimizer.step()
            num_batches += 1

        # Evaluation
        ladder.eval()

        agg_cost /= num_batches
        correct = 0.
        total = 0.
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data[:, 0, :, :].numpy()
            data = data.reshape(data.shape[0], 28 * 28)
            data = torch.FloatTensor(data)
            data, target = Variable(data), Variable(target)
            output = ladder.forward_encoders_clean(data)
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
