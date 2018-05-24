"""
PyTorch implementation of a sequence labeler (POS taggger).

Basic architecture:
 - take words
 - run though bidirectional GRU
 - predict labels one word at a time (left to right), using a recurrent neural network "decoder"

The decoder updates hidden state based on:
 - most recent word
 - the previous action (aka predicted label).
 - the previous hidden state
 
Can it be faster?!?!?!?!?!?

(Adapted from https://gist.github.com/hal3/8c170c4400576eb8d0a8bd94ab231232.)
"""

from __future__ import division
import random
import pickle
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import argparse
from .common import Bench
import gc

def reseed(seed=90210):
    random.seed(seed)
    torch.manual_seed(seed)

reseed()

class Example(object):
    def __init__(self, tokens, labels, n_labels):
        self.tokens = tokens
        self.labels = labels
        self.n_labels = n_labels

def minibatch(data, minibatch_size, reshuffle):
    if reshuffle:
        random.shuffle(data)
    for n in range(0, len(data), minibatch_size):
        yield data[n:n+minibatch_size]

def test_wsj(jit=False, epochs=10):
    jit_tag = '_jit' if jit else ''
    name = 'seqlab{}'.format(jit_tag)
    iter_timer = Bench(name=name, cuda=False, warmup_iters=1)
    print
    print('# test on wsj subset')

    data, n_types, n_labels = pickle.load(open('wsj.pkl', 'rb'))

    d_emb = 50
    d_rnn = 51
    d_hid = 52
    d_actemb = 5

    minibatch_size = 5
    n_epochs = epochs
    preprocess_minibatch = True
    
    embed_word = nn.Embedding(n_types, d_emb)
    gru = nn.GRU(d_emb, d_rnn, bidirectional=True)
    if jit:
        gru = torch.jit.trace(torch.randn(5, 52, 50))(gru)

    embed_action = nn.Embedding(n_labels, d_actemb)
    combine_arh = nn.Linear(d_actemb + d_rnn * 2 + d_hid, d_hid)
    
    initial_h_tensor = torch.Tensor(1, d_hid)
    initial_h_tensor.zero_()
    initial_h = Parameter(initial_h_tensor)
    
    initial_actemb_tensor = torch.Tensor(1, d_actemb)
    initial_actemb_tensor.zero_()
    initial_actemb = Parameter(initial_actemb_tensor)

    policy = nn.Linear(d_hid, n_labels)

    loss_fn = torch.nn.MSELoss(size_average=False)

    optimizer = torch.optim.Adam(
        list(embed_word.parameters()) +
        list(gru.parameters()) +
        list(embed_action.parameters()) +
        list(combine_arh.parameters()) +
        list(policy.parameters()) +
        [initial_h, initial_actemb]
        , lr=0.01)

    for _ in range(n_epochs):
        gc.collect()
        total_loss = 0
        prof = None
        with iter_timer:
        #with torch.autograd.profiler.profile() as prof:
            for batch in minibatch(data, minibatch_size, True):
                optimizer.zero_grad()
                loss = 0

                if preprocess_minibatch:
                    # for efficiency, combine RNN outputs on entire
                    # minibatch in one go (requires padding with zeros,
                    # should be masked but isn't right now)
                    all_tokens = [ex.tokens for ex in batch]
                    if jit:
                        # jit requires fixed length
                        max_length = 52
                    else:
                        max_length = max(map(len, all_tokens))
                    all_tokens = [tok + [0] * (max_length - len(tok)) for tok in all_tokens]
                    all_e = embed_word(Variable(torch.LongTensor(all_tokens), requires_grad=False))
                    [all_rnn_out, _] = gru(all_e)
                
                for ex in batch:
                    N = len(ex.tokens)
                    if preprocess_minibatch:
                        rnn_out = all_rnn_out[0,:,:].view(-1, 1, 2 * d_rnn)
                    else:
                        e = embed_word(Variable(torch.LongTensor(ex.tokens), requires_grad=False)).view(N, 1, -1)
                        [rnn_out, _] = gru(e)
                    prev_h = initial_h  # previous hidden state
                    actemb = initial_actemb  # embedding of previous action
                    output = []
                    for t in range(N):
                        # update hidden state based on most recent
                        # *predicted* action (not ground truth)
                        inputs = [actemb, prev_h, rnn_out[t]]
                        h = F.relu(combine_arh(torch.cat(inputs, 1)))

                        # make prediction
                        pred_vec = policy(h)
                        pred_vec = pred_vec.view(-1)
                        pred = pred_vec.argmin()
                        output.append(pred)

                        # accumulate loss (squared error against costs)
                        truth = torch.ones(n_labels)
                        truth[ex.labels[t]] = 0
                        loss += loss_fn(pred_vec, Variable(truth, requires_grad=False))

                        # cache hidden state, previous action embedding
                        prev_h = h
                        actemb = embed_action(Variable(torch.LongTensor([pred.item()]), requires_grad=False))

                    # print('output=%s, truth=%s' % (output, ex.labels))

                loss.backward()
                total_loss += float(loss)
                optimizer.step()
        if prof is not None:
            print(prof.key_averages())
        print(total_loss)
    return iter_timer
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--jit', action='store_true',
                        help='jit')
    args = parser.parse_args()

    test_wsj(**vars(args))
