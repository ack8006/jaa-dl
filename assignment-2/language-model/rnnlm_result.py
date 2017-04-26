import argparse
import time
import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='../data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1500,
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--encinit', type=str, default='random',
                    help='encoder weight initialization type')
parser.add_argument('--glove_data', type=str, default='../data/glove.6B',
                    help='location of the pretrained glove embeddings')
parser.add_argument('--decinit', type=str, default='random',
                    help='decoder weight initialization type')
parser.add_argument('--weightinit', type=str, default='zeros',
                    help='recurrent hidden weight initialization type')
parser.add_argument('--dropout', type=float, default=0.65,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--optim', type=str, default=None,
                    help='optimizer type')
#parser.add_argument('--tied', action='store_true',
parser.add_argument('--tied', type=int, default=1,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--shuffle', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--vocab', type=int, default=10000,
                    help='size of vocabulary')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=int, default=1,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data, args.vocab, args.shuffle)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def get_glove_embeddings(file_path, corpus, ntoken, nemb):
    file_name = '/glove.6B.{}d.txt'.format(nemb)
    f = open(file_path+file_name, 'r')
    embeddings = torch.nn.init.xavier_uniform(torch.Tensor(ntoken, nemb))
    for line in f:
        split_line = line.split()
        word = split_line[0]
        if word in corpus.dictionary.word2idx:
            embedding = torch.Tensor([float(val) for val in split_line[1:]])
            embeddings[corpus.dictionary.word2idx[word]] = embedding
    return embeddings

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

glove_embeddings = None
if args.encinit == 'glove':
    assert args.emsize in (50, 100, 200, 300)
    glove_embeddings = get_glove_embeddings(args.glove_data, corpus, ntokens, args.emsize)

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout,
                        args.tied, args.encinit, args.decinit, glove_embeddings)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size, args.weightinit)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train(optimizer):
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size, args.weightinit)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        clip_grad_norm(model.parameters(), args.clip)
        if optimizer:
            optimizer.step()
        else:
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000.0 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


model_config = '\t'.join([str(x) for x in (torch.__version__, args.model, args.clip, args.nlayers, args.emsize, args.nhid, args.encinit,
                                    args.decinit, args.weightinit, args.dropout, args.optim, args.lr, args.tied, args.shuffle, ntokens, args.vocab)])

print('Pytorch | RnnType | Clip | #Layers | EmbDim | HiddenDim | EncoderInit | DecoderInit | WeightInit | Dropout | Optimizer| LR | Tied | Shuffle | Ntokens | VocabSize')
print(model_config)

# Loop over epochs.
lr = args.lr
prev_val_loss = None
optimizer = None
if args.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

best_val_perplex = 99999

for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train(optimizer)
    val_loss = evaluate(val_data)
    if math.exp(val_loss) < best_val_perplex:
        best_val_perplex = math.exp(val_loss)
        if args.save != '':
            # save the model
            torch.save(model, args.save)
            # save model state_dict to avoid pytorch version problems
            torch.save(model.state_dict(), args.save + ".state_dict")
            # save config of state_dict which will be needed while loading the model
            with open(args.save + ".state_dict.config", "w") as f:
                f.write(model_config)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f} | best valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss), best_val_perplex))
    print('-' * 89)

    # Anneal the learning rate.
    if prev_val_loss and val_loss > prev_val_loss:
        lr /= 4.0
    prev_val_loss = val_loss


# Run on test data and save the model.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
