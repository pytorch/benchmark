import argparse
import random
import torch

import numpy as np

from .bert_pytorch import parse_args
from .bert_pytorch.trainer import BERTTrainer
from .bert_pytorch.dataset import BERTDataset, WordVocab
from .bert_pytorch.model import BERT
from torch.utils.data import DataLoader
from torchbenchmark.tasks import NLP

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from pathlib import Path

class Model:
    task = NLP.LANGUAGE_MODELING
    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit
        root = str(Path(__file__).parent)
        args = parse_args(args=[
            '--train_dataset', f'{root}/data/corpus.small',
            '--test_dataset', f'{root}/data/corpus.small',
            '--vocab_path', f'{root}/data/vocab.small',
            '--output_path', 'bert.model',
        ]) # Avoid reading sys.argv here
        args.with_cuda = self.device == 'cuda'
        args.script = self.jit
        vocab = WordVocab.load_vocab(args.vocab_path)

        train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                    corpus_lines=args.corpus_lines, on_memory=args.on_memory)
        test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
            if args.test_dataset is not None else None

        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
            if test_dataset is not None else None

        bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

        if args.script:
            bert = torch.jit.script(bert)

        self.trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                                   lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                   with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, debug=args.debug)

        example_batch = next(iter(train_data_loader))
        self.example_inputs = example_batch['bert_input'].to(self.device), example_batch['segment_label'].to(self.device)
        self.is_next = example_batch['is_next'].to(self.device)
        self.bert_label = example_batch['bert_label'].to(self.device)

    def get_module(self):
        return self.trainer.model, self.example_inputs

    def eval(self, niter=1):
        trainer = self.trainer
        for _ in range(niter):
            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = trainer.model.forward(*self.example_inputs)

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            # 2-2. NLLLoss of predicting masked token word
            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            next_loss = trainer.criterion(next_sent_output, self.is_next)
            mask_loss = trainer.criterion(mask_lm_output.transpose(1, 2), self.bert_label)
            loss = next_loss + mask_loss

    def train(self, niter=1):
        trainer = self.trainer
        for _ in range(niter):
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


if __name__ == '__main__':
    for device in ['cpu', 'cuda']:
        for jit in [True, False]:
            print("Testing device {}, JIT {}".format(device, jit))
            m = Model(device=device, jit=jit)
            bert, example_inputs = m.get_module()
            bert(*example_inputs)
            m.train()
            m.eval()
