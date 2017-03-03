from __future__ import print_function

import sys
sys.path.append("/Users/abhishekkadian/Documents/Github/jaa-dl/assignment-1")
sys.path.append("/home/ak6179/jaa-dl/assignment-1/")

import numpy as np
import argparse
import pickle
import random

import torch
from torch.autograd import Variable
from torch.optim import Adam

from encoder import StackedEncoders
from decoder import StackedDecoders


class Ladder(torch.nn.Module):
    def __init__(self, encoder_in, encoder_sizes, decoder_in, decoder_sizes, image_size,
                 encoder_activations, encoder_train_bn_scaling, encoder_bias, noise_std):
        super(Ladder, self).__init__()
        self.se = StackedEncoders(encoder_in, encoder_sizes, encoder_activations,
                                  encoder_train_bn_scaling, encoder_bias, noise_std)
        self.de = StackedDecoders(decoder_in, decoder_sizes, image_size)
        self.bn_input_image = torch.nn.BatchNorm1d(image_size, affine=False)

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


def evaluate_performance(ladder, agg_cost, agg_supervised_cost, agg_unsupervised_cost,
                         num_batches, valid_loader, e, ind_labelled):
    agg_cost_scaled = agg_cost / num_batches
    agg_supervised_cost_scaled = agg_supervised_cost / num_batches
    agg_unsupervised_cost_scaled = agg_unsupervised_cost / num_batches
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

    print("epoch", e + 1, ", ind_labelled:", ind_labelled,
          ", total cost:", "{:.4f}".format(agg_cost_scaled),
          ", supervised cost:", "{:.4f}".format(agg_supervised_cost_scaled),
          ", unsupervised cost:", "{:.4f}".format(agg_unsupervised_cost_scaled),
          ", validation accuracy:", correct / total)
    print("")


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
    with open("../../data/train_unlabeled.p") as f:
        unlabeled_dataset = pickle.load(f)
    unlabeled_dataset.train_labels = torch.LongTensor(
        [-1 for x in range(unlabeled_dataset.train_data.size()[0])])
    with open("../../data/validation.p") as f:
        valid_dataset = pickle.load(f)
    print("===========================")

    loader_kwargs = {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    train_labelled_data = []
    for data, target in train_loader:
        train_labelled_data.append((data, target))

    encoder_in = 28 * 28
    decoder_in = 10
    encoder_sizes = [1000, 500, 250, 250, 250, decoder_in]
    decoder_sizes = [250, 250, 250, 500, 1000, encoder_in]
    unsupervised_costs_lambda = [0.1, 0.1, 0.1, 0.1, 0.1, 20., 2000.]

    encoder_activations = ["relu", "relu", "relu", "relu", "relu", "softmax"]
    # TODO: Verify whether you need affine for relu.
    encoder_train_bn_scaling = [False, False, False, False, False, True]
    encoder_bias = [False, False, False, False, False, False]

    ladder = Ladder(encoder_in, encoder_sizes, decoder_in, decoder_sizes, encoder_in,
                    encoder_activations, encoder_train_bn_scaling, encoder_bias, noise_std)

    optimizer = Adam(ladder.parameters(), lr=0.002)
    loss_labelled = torch.nn.CrossEntropyLoss()
    loss_unsupervised = torch.nn.MSELoss()

    print("")
    print("=======NETWORK=======")
    print(ladder)
    print("=====================")

    print("")
    print("=====================")
    print("TRAINING\n")

    # TODO: Add annealing of learning rate after 100 epochs

    for e in range(epochs):
        agg_cost = 0.
        agg_supervised_cost = 0.
        agg_unsupervised_cost = 0.
        num_batches = 0

        # Training
        ladder.train()

        ind_labelled = 0

        for batch_idx, (unlabeled_data, unlabeled_target) in enumerate(unlabeled_loader):

            # Labelled data
            if ind_labelled == len(train_labelled_data):
                # reset counter
                random.shuffle(train_labelled_data)
                ind_labelled = 0
            labelled_data = train_labelled_data[ind_labelled][0]
            labelled_target = train_labelled_data[ind_labelled][1]
            ind_labelled += 1

            labelled_data_size = labelled_data.size()[0]

            data = np.concatenate((labelled_data.numpy(), unlabeled_data.numpy()), axis=0)
            target = np.concatenate((labelled_target.numpy(), unlabeled_target.numpy()), axis=0)

            data = torch.FloatTensor(data)
            target = torch.LongTensor(target)

            data = data[:,0,:,:].numpy()
            data = data.reshape(data.shape[0], 28 * 28)
            data = torch.FloatTensor(data)
            # TODO: Hold off on this, things should work right now because LongTensor is only used for cost.
            # TODO: Change from LongTensor to FloatTensor. Autograd has a bug with LongTensor.
            data, target = Variable(data), Variable(target)

            # Pass data through the network

            optimizer.zero_grad()

            # do a noisy pass
            output_noise = ladder.forward_encoders_noise(data)
            tilde_z_layers = ladder.get_encoders_tilde_z(reverse=True)

            # do a clean pass
            _ = ladder.forward_encoders_clean(data)
            z_pre_layers = ladder.get_encoders_z_pre(reverse=True)
            z_layers = ladder.get_encoders_z(reverse=True)

            # pass through decoders
            hat_z_layers = ladder.forward_decoders(tilde_z_layers, output_noise)

            z_pre_layers.append(data)

            # TODO: Verify if you have to batch-normalize the bottom-most layer also

            # batch normalizing the image also for cost comparison
            bn_data = ladder.bn_input_image(data)
            z_layers.append(bn_data)

            # batch normalize using mean, var of z_pre
            bn_hat_z_layers = ladder.decoder_bn_hat_z_layers(hat_z_layers, z_pre_layers)

            assert len(z_layers) == len(bn_hat_z_layers)

            # calculate costs
            cost_supervised = loss_labelled.forward(output_noise[:labelled_data_size], target[:labelled_data_size])
            cost_unsupervised = 0.
            for cost_lambda, z, bn_hat_z in zip(unsupervised_costs_lambda, z_layers, bn_hat_z_layers):
                c = cost_lambda * loss_unsupervised.forward(bn_hat_z, z)
                cost_unsupervised += c

            # backprop
            cost = cost_supervised + cost_unsupervised
            cost.backward()

            agg_cost += cost.data[0]
            agg_supervised_cost += cost_supervised.data[0]
            agg_unsupervised_cost += cost_unsupervised.data[0]

            optimizer.step()

            num_batches += 1

            if ind_labelled == len(train_labelled_data):
                # Evaluation
                ladder.eval()
                evaluate_performance(ladder, agg_cost, agg_supervised_cost, agg_unsupervised_cost,
                                     num_batches, valid_loader, e, ind_labelled)
                # reset costs
                agg_cost = 0.
                agg_supervised_cost = 0.
                agg_unsupervised_cost = 0.
                num_batches = 0
                ladder.train()

        # Evaluation
        ladder.eval()
        evaluate_performance(ladder, agg_cost, agg_supervised_cost, agg_unsupervised_cost,
                             num_batches, valid_loader, e, ind_labelled)
        ladder.train()

    print("=====================\n")

    print("Done :)")


if __name__ == "__main__":
    main()
