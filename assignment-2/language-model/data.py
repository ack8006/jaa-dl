import os
from collections import Counter
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, vocab_size=10000):
        self.dictionary = Dictionary()
        self.vocab_size = vocab_size
        self.gen_vocab(os.path.join(path, 'train.txt'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))


    def gen_vocab(self, path):
        assert os.path.exists(path)
        with open(path, 'r') as f:
            all_words = []
            for line in f:
                for word in line.split() + ['<eos>']:
                    all_words.append(word)
            for word, _ in Counter(all_words).most_common(self.vocab_size):
                self.dictionary.add_word(word)


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        token_list = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        token_list.append(self.dictionary.word2idx[word])
                    else:
                        token_list.append(self.dictionary.word2idx['<unk>'])
        return torch.LongTensor(token_list)
