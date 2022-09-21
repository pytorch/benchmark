import argparse
import random
import torch

import numpy as np

from torchbenchmark.util.env_check import set_random_seed

from .bert_pytorch import parse_args
from .bert_pytorch.trainer import BERTTrainer
from .bert_pytorch.dataset import BERTDataset, WordVocab
from .bert_pytorch.model import BERT
from torch.utils.data import DataLoader
from torch.utils._pytree import tree_map
import typing

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from pathlib import Path
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP

import io

class CorpusGenerator(io.TextIOBase):
    """
    Class to Generate Random Corpus in Lieu of Using Fixed File Data.

    Model is written to consume large fixed corpus but for purposes
    of benchmark its sufficient to generate nonsense corpus with
    similar distribution.

    Corpus is sentence pairs. Vocabulary words are simply numbers and
    sentences are each 1-4 words.

    Deriving from TextUIBase allows object to participate as a text
    file.
    """


    def __init__(self, words, lines):
        self.lines_read = 0
        self.lines = lines
        self.words = words

    def reset(self):
        self.lines_read = 0

    def readable(self):
        return self.lines <= self.lines_read

    def readline(self):

        self.lines_read = self.lines_read + 1
        if (self.lines_read > self.lines):
          return ""

        newline = ""
        for j in range(random.randrange(1,4)):
            newline += str(random.randrange(self.words)) + " "

        newline += "\\t "

        for j in range(random.randrange(1,4)):
            newline += str(random.randrange(self.words)) + " "

        newline += "\n"

        #print(newline)

        return newline

class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE = 16
    DEFAULT_EVAL_BSIZE = 16

    def __init__(self, test, device, batch_size=None, jit=False, extra_args=[]):
        if device == "cpu":
            self.DEFAULT_EVAL_BSIZE = max(1, int(self.DEFAULT_EVAL_BSIZE / 8))
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        debug_print = False
        root = str(Path(__file__).parent)
        args = parse_args(args=[
            '--train_dataset', f'{root}/data/corpus.small',
            '--test_dataset', f'{root}/data/corpus.small',
            '--vocab_path', f'{root}/data/vocab.small',
            '--output_path', 'bert.model',
        ]) # Avoid reading sys.argv here
        args.with_cuda = self.device == 'cuda'
        args.script = self.jit
        args.on_memory = True

        # Example effect of batch size on eval time(ms)
        # bs     cpu       cuda
        # 1      330       15.5
        # 2      660       15.5
        # 4     1200       15.2
        # 8     2200       20
        # 16    4350       33
        # 32    8000       58
        #
        # Issue is that with small batch sizes the gpu is starved for work.
        # Ideally doubling work would double execution time.

        # parameters for work size, these were chosen to provide a profile
        # that matches processing of an original trained en-de corpus.
        args.batch_size = self.batch_size
        vocab_size = 20000
        args.corpus_lines = 50000

        # generate random corpus from parameters
        set_random_seed()
        vocab = WordVocab(CorpusGenerator(vocab_size, args.corpus_lines))

        #with open(args.train_dataset, "r", encoding="utf-8") as f:
        #  vocab = WordVocab(f)
        #vocab = WordVocab.load_vocab(args.vocab_path)

        if debug_print:
            print("seq_len:")
            print(args.seq_len)
            print("batch size:")
            print(args.batch_size)
            print("layers")
            print(args.layers)
            print("args hidden:")
            print(args.hidden)
            print("len vocab:")
            print(len(vocab))
            print(type(vocab))

        set_random_seed()
        train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                    corpus_lines=args.corpus_lines, on_memory=args.on_memory, generator = CorpusGenerator(vocab_size, args.corpus_lines))

        set_random_seed()
        test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory, generator = CorpusGenerator(vocab_size, args.corpus_lines)) \
            if args.test_dataset is not None else None

        set_random_seed()

        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
            if test_dataset is not None else None

        bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

        trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                                   lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                   with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, debug=args.debug)

        example_batch = next(iter(train_data_loader))
        self.example_inputs = example_batch['bert_input'].to(self.device)[:self.batch_size], example_batch['segment_label'].to(self.device)[:self.batch_size]
        self.is_next = example_batch['is_next'].to(self.device)[:self.batch_size]
        self.bert_label = example_batch['bert_label'].to(self.device)[:self.batch_size]
        self.model = trainer

    def get_module(self):
        return self.model.bert, self.example_inputs

    def enable_fp16_half(self):
        self.model.model.half()
        self.example_inputs = tree_map(
            lambda x: x.to(torch.float16)
            if getattr(x, "dtype", None) == torch.float32
            or getattr(x, "dtype", None) == torch.float64
            else x,
            self.example_inputs,
        )

    def set_module(self, new_model):
        self.model.bert = new_model

    def eval(self) -> typing.Tuple[torch.Tensor]:
        model = self.model
        # 1. forward the next_sentence_prediction and masked_lm model
        next_sent_output, mask_lm_output = model.model.forward(*self.example_inputs)

        # 2-1. NLL(negative log likelihood) loss of is_next classification result
        # 2-2. NLLLoss of predicting masked token word
        # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
        next_loss = model.criterion(next_sent_output, self.is_next)
        mask_loss = model.criterion(mask_lm_output.transpose(1, 2), self.bert_label)
        loss = next_loss + mask_loss
        return (next_sent_output, mask_lm_output)

    def train(self):
        trainer = self.model
        # 1. forward the next_sentence_prediction and masked_lm model
        next_sent_output, mask_lm_output = trainer.model.forward(*self.example_inputs)

        # 2-1. NLL(negative log likelihood) loss of is_next classification result
        # 2-2. NLLLoss of predicting masked token word
        # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
        next_loss = trainer.criterion(next_sent_output, self.is_next)
        mask_loss = trainer.criterion(mask_lm_output.transpose(1, 2), self.bert_label)
        loss = next_loss + mask_loss

        # 3. backward and optimization only in train
        trainer.optim_schedule.zero_grad()
        loss.backward()
        trainer.optim_schedule.step_and_update_lr()
