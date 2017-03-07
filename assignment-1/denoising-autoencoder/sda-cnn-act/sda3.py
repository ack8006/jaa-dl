from  __future__ import print_function, division

#import sys
#sys.path.append('/Users/Alex/GitHub/jaa-dl/assignment-1/')
import pickle

import numpy as np

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import autoencoder as ae


class SDA(torch.nn.Module):
    """Stacked Denoising Autoencoder

    reference: http://www.deeplearning.net/tutorial/SdA.html,
               http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf
    """
    def __init__(self, d_input, d_hidden_autoencoders, d_out,
                 corruptions, batch_size, pre_lr=0.001, ft_lr=0.1):
        super(SDA, self).__init__()
        self.d_input = d_input
        self.d_hidden_autoencoders = list(d_hidden_autoencoders)
        self.d_out = d_out
        self.corruptions = corruptions
        self.batch_size = batch_size
        self.pre_lr = pre_lr
        self.ft_lr = ft_lr

        # Create one sequential module containing all autoencoders and logistic layer
        self.sequential = torch.nn.Sequential()

        # Create the Autoencoders
        self.autoencoders_ref = []
        for i, (d, c) in enumerate(zip(d_hidden_autoencoders, corruptions)):
            if i == 0:
                curr_input = d_input
            else:
                curr_input = d_hidden_autoencoders[i - 1]
            dna = ae.Autoencoder(curr_input, d, batch_size, corruption=c)
            self.autoencoders_ref.append("autoencoder_" + str(i))
            self.sequential.add_module(self.autoencoders_ref[-1], dna)

        # Create the Logistic Layer
        self.sequential.add_module("top_linear1", torch.nn.Linear(d_hidden_autoencoders[-1], d_out, bias=True))
        self.sequential.top_linear1 = torch.nn.Linear(d_hidden_autoencoders[-1], d_out, bias=True)
        self.sequential.top_linear1.weight.data = torch.zeros(self.sequential.top_linear1.weight.data.size())
        self.sequential.top_linear1.bias.data = torch.zeros(d_out)
        self.sequential.add_module("softmax", torch.nn.LogSoftmax())

    def pretrain(self, train_loader, pt_epochs, verbose=True):
        # Pre-train 1 autoencoder at a time
        for i, ae_re in enumerate(self.autoencoders_ref):
            ae = getattr(self.sequential, ae_re)
            optimizer = Adam(ae.parameters())
            for it in xrange(pt_epochs):
                agg_cost = 0.0
                for ind, (data, _) in enumerate(train_loader):
                    data = Variable(data)
                    clean_encode = data
                    for prev_i in xrange(i):
                        prev_ae = getattr(self.sequential, self.autoencoders_ref[prev_i])
                        clean_encode = prev_ae.encode(clean_encode, add_noise=False)
                    optimizer.zero_grad()
                    z = ae.encode(clean_encode, add_noise=True)
                    z = ae.decode(z)
                    loss = -torch.sum(clean_encode * torch.log(z) + (1.0-clean_encode) * torch.log(1.0-z), 1)
                    cost = torch.mean(loss)
                    cost.backward()
                    optimizer.step()
                    agg_cost += cost
                agg_cost = agg_cost / len(train_loader)
                if verbose:
                    print("Pre-training Autoencoder:", i, "Epoch:", it, "Cost:", agg_cost.data[0])

    def forward(self, x):
        t = self.sequential.forward(x)
        return t

    def ae_decoder_forward(self, x):
        for i, ae_re in enumerate(self.autoencoders_ref):
            ae = getattr(self.sequential, ae_re)
            x = ae.encode(x, add_noise=False)
            if i == (len(self.autoencoders_ref)-1):
                x = ae.decode(x)
        return x

    # def finetune(self, train_X, train_y, valid_X, valid_y, valid_actual_size, ft_epochs, verbose=True):
    def finetune(self, train_loader, valid_loader, ft_epochs, verbose=True):
        # n = train_X.data.size()[0]
        # num_batches = n / self.batch_size
        # n_v = valid_X.data.size()[0]
        # num_batches_v = n_v / self.batch_size

        #optimizer = SGD(self.parameters(), lr=self.ft_lr)
        optimizer = Adam(self.parameters())#, lr=self.ft_lr)
        loss = torch.nn.NLLLoss()
        loss = torch.nn.CrossEntropyLoss(size_average=True)

        for it in xrange(ft_epochs):
            agg_cost = 0.0
            for ind, (data, value) in enumerate(train_loader):
                x = Variable(data, requires_grad=False)
                y = Variable(value, requires_grad=False)
                optimizer.zero_grad()

                fx = self.forward(x)

                cost = loss.forward(fx, y)
                agg_cost += cost
                cost.backward()
                optimizer.step()

            total_val = 0.0
            total_cor = 0
            for ind, (data, value) in enumerate(valid_loader):
                x = Variable(data, requires_grad=False)
                output = self.forward(x)
                predY = output.data.numpy().argmax(axis=1)
                total_cor += np.sum(predY == value.numpy())
                total_val += len(value)

            validation_accuracy = (100. * (total_cor/total_val))

            print('It: {0}, Cost: {1}, val_acc = {2}'.format(str(it), str(agg_cost.data[0]), str(validation_accuracy)))



        # for ef in range(ft_epochs):
        #     agg_cost = 0
        #     for k in range(num_batches):
        #         start, end = k * self.batch_size, (k + 1) * self.batch_size
        #         bX = train_X[start:end]
        #         by = train_y[start:end]
        #         optimizer.zero_grad()
        #         p = self.forward(bX)
        #         cost = loss.forward(p, by)
        #         agg_cost += cost
        #         cost.backward()
        #         optimizer.step()
        #     agg_cost /= num_batches
        #     preds = np.zeros((n_v, self.d_out))

            # Calculate accuracy on Validation set
            # for k in range(num_batches_v):
            #     start, end = k * self.batch_size, (k + 1) * self.batch_size
            #     bX = valid_X[start:end]
            #     p = self.forward(bX).data.numpy()
            #     preds[start:end] = p
            # correct = 0
            # for actual, prediction in zip(valid_y[:valid_actual_size], preds[:valid_actual_size]):
            #     ind = np.argmax(prediction)
            #     actual = actual.data.numpy()
            #     print('Actual ', actual)
            #     print('Prediction ', prediction)
            #     print('Ind ', ind)
            #     print()
            #     if ind == actual:
            #         correct += 1

            # if verbose:
            #     print("Fine-tuning Epoch:", ef, "Cost:", agg_cost.data[0],
            #           "Validation Accuracy:", "{0:.4f}".format(correct / float(valid_actual_size)))


def main():
    # torch.manual_seed(42)

    PRETRAIN_EPOCHS = 20
    FINETUNE_EPOCHS = 30
    BATCH_SIZE = 50
    DATA_PATH = '/Users/Alex/GitHub/jaa-dl/assignment-1/data/'


    train_data = pickle.load(open(DATA_PATH+'generated_train_data_norm.p', 'rb'))
    train_data = np.array([x.flatten() for x in train_data[:,0,:,:]])
    train_data = torch.from_numpy(train_data).float()
    train_label = pickle.load(open(DATA_PATH+'generated_train_labels.p', 'rb'))
    train_label = torch.from_numpy(train_label).long()

    valid_data = pickle.load(open(DATA_PATH+'generated_valid_data_norm.p', 'rb'))
    valid_data = np.array([x.flatten() for x in valid_data[:,0,:,:]])
    valid_data = torch.from_numpy(valid_data).float()
    valid_label = pickle.load(open(DATA_PATH+'generated_valid_labels.p', 'rb'))
    valid_label = torch.from_numpy(valid_label).long()

    print('Creating Data Loaders')
    train_loader = DataLoader(TensorDataset(train_data, train_label),
                                batch_size = BATCH_SIZE,
                                shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_data, valid_label),
                                batch_size = len(valid_data),
                                shuffle=False)

    # Load data
    # trX = pickle.load(open(DATA_PATH+'generated_train_data_norm.p', 'rb'))
    # trX = np.array([x.flatten() for x in trX[:,0,:,:]])
    # trY = pickle.load(open(DATA_PATH+'generated_train_labels.p', 'rb'))
    # teX = pickle.load(open(DATA_PATH+'generated_valid_data_norm.p', 'rb'))
    # teX = np.array([x.flatten() for x in teX[:,0,:,:]])
    # teY = pickle.load(open(DATA_PATH+'generated_valid_labels.p','rb'))

    # #trX = np.array([x.flatten() for x in trX])
    # #teX = np.array([x.flatten() for x in teX])
    # trX = Variable(torch.from_numpy(trX).float())
    # teX = Variable(torch.from_numpy(teX).float())
    # trY = Variable(torch.from_numpy(trY).long())
    # teY = Variable(torch.from_numpy(teY).long())

    

    # Pad the validation set
    #actual_size = teX.size()[0]
    #padded_size = (actual_size / batch_size + 1) * batch_size
    #teX_padded = Variable(torch.FloatTensor(padded_size, teX.size()[1]))
    #teY_padded = Variable(torch.LongTensor(padded_size) * 0)
    #teX_padded[:actual_size] = teX
    #teY_padded[:actual_size] = teY

    sda = SDA(d_input=784,
              d_hidden_autoencoders=[1000, 1000, 1000],
              d_out=10,
              corruptions=[.1, .2, .3],
              batch_size=BATCH_SIZE)

    # sda.pretrain(trX, pt_epochs = PRETRAIN_EPOCHS)
    # sda.pretrain(train_loader, pt_epochs = PRETRAIN_EPOCHS)

    # with open('sda_pretrained.model', 'w') as f:
    #     torch.save(sda, f)
    sda = torch.load('sda_pretrained.model')
    print('Model Loaded')

    # sda.finetune(trX, trY, teX_padded, teY_padded,
    #              valid_actual_size=actual_size, ft_epochs = FINETUNE_EPOCHS)
    sda.finetune(train_loader, valid_loader, ft_epochs = FINETUNE_EPOCHS)



if __name__ == "__main__":
    main()




