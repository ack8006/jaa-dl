import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, weight_init='random', init_val=0.1):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=False)
        self.decoder = nn.Linear(nhid, ntoken)

        self.weight_init = weight_init
        self.init_weights(init_val)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self, init_val):
        bias_init = 0.1
        init.constant(self.decoder.bias, bias_init)
        # self.decoder.bias.data.fill_(0)
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.weight.data.uniform_(-initrange, initrange)

        if self.weight_init == 'random':
            init.uniform(self.encoder.weight, -init_val, init_val)
            init.uniform(self.decoder.weight, -init_val, init_val)
        elif self.weight_init == 'uniform':
            init.constant(self.encoder.weight, init_val)
            init.constant(self.decoder.weight, init_val)
        elif self.weight_init == 'xavier_n':
            init.xavier_normal(self.encoder.weight)
            init.xavier_normal(self.decoder.weight)
        elif self.weight_init == 'xavier_u':
            init.xavier_uniform(self.encoder.weight)
            init.xavier_uniform(self.decoder.weight)
        else:
            raise Exception('{} is not a valid weight initialization type, please use [random, xavier_n, or xavier_u]'.format(self.weight_init))

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
