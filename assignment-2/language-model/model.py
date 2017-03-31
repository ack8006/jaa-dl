import functools
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout, tie_weights, 
                    encoder_init, decoder_init, glove_embeddings):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.decoder_init = decoder_init
        self.encoder_init = encoder_init

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_val = 0.1
        self.init_weights(glove_embeddings)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self, glove_embeddings):
        bias_init_val = 0
        init.constant(self.decoder.bias, bias_init_val)
        init_types = {'random':functools.partial(init.uniform, a=-self.init_val, b=self.init_val),
                        'constant': functools.partial(init.constant, val=self.init_val),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}
        if self.encoder_init == 'glove':
            self.encoder.weight.data = glove_embeddings
        else:
            init_types[self.encoder_init](self.encoder.weight)
        init_types[self.decoder_init](self.decoder.weight)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, weight_init):
        init_types = {'random':functools.partial(init.uniform, a=-self.init_val, b=self.init_val),
                        'constant': functools.partial(init.constant, val=self.init_val),
                        'zeros': functools.partial(init.constant, val=0.0),
                        'xavier_n': init.xavier_normal,
                        'xavier_u': init.xavier_uniform,
                        'orthogonal': init.orthogonal}

        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(init_types[weight_init](weight.new(self.nlayers, bsz, self.nhid))),
                    Variable(init_types[weight_init](weight.new(self.nlayers, bsz, self.nhid))))
        else:
            return Variable(init_types[weight_init](weight.new(self.nlayers, bsz, self.nhid)))
