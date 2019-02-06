import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import *

# This file is superceded by sequence_labeler.py.


class POSTagger(Benchmark):
    """Based on code from https://gist.github.com/hal3/8c170c4400576eb8d0a8bd94ab231232 (by Hal Daume III)."""
    goal_time = 2
    sequence_lengths = [13, 32, 35, 12, 16, 10, 21, 25, 23, 17, 22, 36, 23, 42,
                        52, 19, 18, 13, 34, 24, 31, 10, 21, 16, 24, 18, 33, 15,
                        21, 19, 16, 12, 18, 15, 15, 20, 23, 16, 26, 19, 31, 4,
                        18, 18, 31, 26, 5, 29, 16, 19]

    default_params = dict(
        embedding_size=50,
        rnn_size=51,
        hidden_size=52,
        action_embedding_size=5,
        num_input_tokens=32,
        num_labels=32,
        minibatch_size=5,
        preprocess_minibatch=True,
        cuda=False)
    params = make_params(preprocess_minibatch=over(True, False))

    def prepare(self, p):
        self.embed_word = nn.Embedding(p.num_input_tokens, p.embedding_size)
        self.gru = nn.GRU(p.embedding_size, p.rnn_size, bidirectional=True)
        # Decoder modules
        self.embed_action = nn.Embedding(p.num_labels, p.action_embedding_size)
        self.combine_arh = nn.Linear(p.action_embedding_size + p.rnn_size * 2 + p.hidden_size, p.hidden_size)
        self.policy = nn.Linear(p.hidden_size, p.num_labels)
        self.loss_fn = torch.nn.MSELoss(size_average=False)

        if p.cuda:
            for module in [self.embed_word, self.gru, self.embed_action, self.combine_arh, self.policy, self.loss_fn]:
                module.cuda()
            self.LongTensor = torch.cuda.LongTensor

            def cast(t):
                return t.cuda()
        else:
            self.LongTensor = torch.LongTensor

            def cast(t):
                return t
        self.cast = cast

        self.initial_h = Variable(cast(torch.zeros(1, p.hidden_size)), requires_grad=True)
        self.initial_actemb = Variable(cast(torch.zeros(1, p.action_embedding_size)), requires_grad=True)

        # Sample input tokens and labels for each sentence
        tokens = [cast(torch.LongTensor(l).random_(p.num_input_tokens)) for l in self.sequence_lengths]
        # NOTE: we don't cast labels to CUDA, because they're only used for indexing
        labels = [torch.LongTensor(l).random_(p.num_labels) for l in self.sequence_lengths]
        # Batch sentences in groups of minibatch_size
        self.batches = [(tokens[n:n + p.minibatch_size], labels[n:n + p.minibatch_size])
                        for n in range(0, len(tokens), p.minibatch_size)]

    def time_pos_tagger(self, p):
        for token_batch, label_batch in self.batches:

            if p.preprocess_minibatch:
                # Requires padding with zeros, should be masked but isn't right now
                max_length = max(map(len, token_batch))
                # Pad the sequences
                all_tokens = self.LongTensor(p.minibatch_size, max_length).zero_()
                for i, tokens in enumerate(token_batch):
                    all_tokens[i, :len(tokens)] = tokens
                all_e = self.embed_word(Variable(all_tokens))
                all_rnn_out, _ = self.gru(all_e)

            loss = 0
            for i, (tokens, labels) in enumerate(zip(token_batch, label_batch)):
                seq_len = len(tokens)
                if p.preprocess_minibatch:
                    rnn_out = all_rnn_out[i].view(-1, 1, 2 * p.rnn_size)
                else:
                    e = self.embed_word(Variable(tokens)).view(seq_len, 1, -1)
                    rnn_out, _ = self.gru(e)

                # The first code block inside the loop is a simple RNN cell that is fed with:
                # * embedding of previous prediction (min over outputs) <- this makes hand-coding this loop necessary
                # * previous hidden state
                # * pre-processed input for the current step
                # Then, we compute the loss, and the prediction embedding for the next step.
                prev_h = self.initial_h
                actemb = self.initial_actemb
                for t in range(seq_len):
                    # Make a prediction
                    inputs = [actemb, prev_h, rnn_out[t]]
                    h = F.relu(self.combine_arh(torch.cat(inputs, 1)))
                    pred_vec = self.policy(h)
                    _, pred = pred_vec.squeeze(0).min(0)

                    # Accumulate loss
                    truth = torch.ones(p.num_labels)
                    truth[labels[t]] = 0
                    loss += self.loss_fn(pred_vec, Variable(self.cast(truth)))

                    # Prepare hidden state and prediction embedding for the next step
                    prev_h = h
                    actemb = self.embed_action(pred)

            loss.backward()
