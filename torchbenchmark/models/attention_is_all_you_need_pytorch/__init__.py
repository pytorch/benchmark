from argparse import Namespace
import math
import time
import os
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchbenchmark.util.torchtext_legacy.field import Field
from torchbenchmark.util.torchtext_legacy.data import Dataset
from torchbenchmark.util.torchtext_legacy.iterator import BucketIterator
from torchbenchmark.util.torchtext_legacy.translation import TranslationDataset

from .transformer import Constants
from .transformer.Models import Transformer
from .transformer.Optim import ScheduledOptim
from .train import prepare_dataloaders, cal_performance, patch_src, patch_trg
import random
import numpy as np
from pathlib import Path
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

class Model(BenchmarkModel):
    task = NLP.TRANSLATION
    optimized_for_inference = True

    def _create_transformer(self):
        transformer = Transformer(
            self.opt.src_vocab_size,
            self.opt.trg_vocab_size,
            src_pad_idx=self.opt.src_pad_idx,
            trg_pad_idx=self.opt.trg_pad_idx,
            trg_emb_prj_weight_sharing=self.opt.proj_share_weight,
            emb_src_trg_weight_sharing=self.opt.embs_share_weight,
            d_k=self.opt.d_k,
            d_v=self.opt.d_v,
            d_model=self.opt.d_model,
            d_word_vec=self.opt.d_word_vec,
            d_inner=self.opt.d_inner_hid,
            n_layers=self.opt.n_layers,
            n_head=self.opt.n_head,
            dropout=self.opt.dropout).to(self.device)
        
        return transformer

    def _preprocess(self, data_iter):
        preloaded_data = []
        for d in data_iter:
            src_seq = patch_src(d.src, self.opt.src_pad_idx).to(self.device)
            trg_seq, gold = map(lambda x: x.to(self.device), patch_trg(d.trg, self.opt.trg_pad_idx))
            preloaded_data.append((src_seq, trg_seq, gold))
        return preloaded_data

    # Original batch size 256, hardware platform unknown
    # Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/README.md?plain=1#L83
    def __init__(self, test, device, jit=False, train_bs=256, eval_bs=32, extra_args=[]):
        super().__init__()
        self.device = device
        self.jit = jit
        self.test = test
        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.extra_args = extra_args

        root = os.path.join(str(Path(__file__).parent), ".data")
        self.opt = Namespace(**{
            'batch_size': train_bs,
            'eval_batch_size': eval_bs,
            'd_inner_hid': 2048,
            'd_k': 64,
            'd_model': 512,
            'd_word_vec': 512,
            'd_v': 64,
            'data_pkl': f'{root}/m30k_deen_shr.pkl',
            'debug': '',
            'dropout': 0.1,
            'embs_share_weight': False,
            'epoch': 1,
            'label_smoothing': False,
            'log': None,
            'n_head': 8,
            'n_layers': 6,
            'n_warmup_steps': 128,
            'cuda': True,
            'proj_share_weight': False,
            'save_mode': 'best',
            'save_model': None,
            'script': False,
            'train_path': None,
            'val_path': None,
        })

        train_data, test_data = prepare_dataloaders(self.opt, self.device)

        transformer = self._create_transformer()
        self.eval_model = self._create_transformer()
        self.eval_model.eval()

        self.train_data_loader = self._preprocess(train_data)
        self.eval_data_loader = self._preprocess(test_data)
        src_seq, trg_seq, gold = self.train_data_loader[0]
        example_inputs = (src_seq, trg_seq)
        if self.jit:
            if hasattr(torch.jit, '_script_pdt'):
                transformer = torch.jit._script_pdt(transformer, example_inputs = [example_inputs, ])
                self.eval_model = torch.jit._script_pdt(self.eval_model, example_inputs = [example_inputs, ])
            else:
                transformer = torch.jit.script(transformer, example_inputs = [example_inputs, ])
                self.eval_model = torch.jit.script(self.eval_model, example_inputs = [example_inputs, ])
            self.eval_model = torch.jit.optimize_for_inference(self.eval_model)
        self.module = transformer
        self.optimizer = ScheduledOptim(
            optim.Adam(self.module.parameters(), betas=(0.9, 0.98), eps=1e-09),
            2.0, self.opt.d_model, self.opt.n_warmup_steps)

    def get_module(self):
        for (src_seq, trg_seq, gold) in self.train_data_loader:
            return self.module, (*(src_seq, trg_seq), )

    def eval(self, niter=1):
        self.module.eval()
        for _, (src_seq, trg_seq, gold) in zip(range(niter), self.eval_data_loader):
            self.eval_model(*(src_seq, trg_seq))

    def train(self, niter=1):
        self.module.train()
        for _, (src_seq, trg_seq, gold) in zip(range(niter), self.train_data_loader):
            self.optimizer.zero_grad()
            example_inputs = (src_seq, trg_seq)
            pred = self.module(*example_inputs)
            loss, n_correct, n_word = cal_performance(
                pred, gold, self.opt.trg_pad_idx, smoothing=self.opt.label_smoothing)
            loss.backward()
            self.optimizer.step_and_update_lr()
