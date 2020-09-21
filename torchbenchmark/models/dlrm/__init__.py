from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools
# import bisect
# import shutil
import time
import json
# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import onnx

# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter

# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag
# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

import sklearn.metrics
import sys
from torch.optim.lr_scheduler import _LRScheduler
from dlrm_s_pytorch import DLRM_Net,LRPolicyScheduler
from argparse import Namespace

exc = getattr(builtins, "IOError", "FileNotFoundError")

class Model:
    def __init__(self, device='cpu', jit=False):
        self.device = device
        self.jit = jit
        self.opt = Namespace(**{
            'm_spa' : None,
            'ln_emb': None,
            'ln_bot': None,
            'ln_top': None,
            'arch_interaction_op': "dot",
            'arch_interaction_itself': False,
            'sigmoid_bot': -1,
            'sigmoid_top': -1,
            'sync_dense_params': True,
            'loss_threshold': 0.0,
            'ndevices': -1,
            'qr_flag': False,
            'qr_operation': "mult",
            'qr_collisions': 0,
            'qr_threshold': 200,
            'md_flag': False,
            'md_threshold': 200,
            'md_temperature': 0.3,
            'activation_function': "relu",
            'loss_function': "bce",
            'loss_weights': "1.0-1.0",
            'loss_threshold': 0.0,
            'round_targets': False,
            'data_size': 6,
            'num_batches': 0,
            'data_generation': "random",
            'data_trace_file': "./input/dist_emb_j.log",
            'raw_data_file': "",
            'processed_data_file': "",
            'data_randomize': "total",
            'data_trace_enable_padding': False,
            'max_ind_range': -1,
            'num_workers': 0,
            'memory_map': False,
            'data_sub_sample_rate': 0.0,
            'learning_rate': 0.01,
            'lr_num_warmup_steps': 0,
            'lr_decay_start_step': 0,
            'lr_num_decay_steps': 0,
            'arch_embedding_size': "4-3-2",
            'arch_mlp_bot': "4-3-2",
            'arch_mlp_top': "4-2-1",
            'mini_batch_size': 2,
            'num_indices_per_lookup': 10,
            'num_indices_per_lookup_fixed': True,
            'numpy_rand_seed': 123,
            'arch_sparse_feature_size': 2,
        })

        if self.jit:
            raise NotImplementedError()

        ### some basic setup ###
        np.random.seed(self.opt.numpy_rand_seed)
        torch.manual_seed(self.opt.numpy_rand_seed)

        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.opt.numpy_rand_seed)
            torch.backends.cudnn.deterministic = True

        ### prepare training data ###
        self.opt.ln_bot = np.fromstring(self.opt.arch_mlp_bot, dtype=int, sep="-")

        # input and target at random
        self.opt.ln_emb = np.fromstring(self.opt.arch_embedding_size, dtype=int, sep="-")
        self.opt.m_den = self.opt.ln_bot[0]
        train_data, self.train_ld = dp.make_random_data_and_loader(self.opt, self.opt.ln_emb, self.opt.m_den)
        self.opt.nbatches = len(self.train_ld)

        self.opt.m_spa = self.opt.arch_sparse_feature_size
        num_fea = self.opt.ln_emb.size + 1  # num sparse + num dense features
        m_den_out = self.opt.ln_bot[self.opt.ln_bot.size - 1]
        if self.opt.arch_interaction_op == "dot":
            # approach 1: all
            # num_int = num_fea * num_fea + m_den_out
            # approach 2: unique
            if self.opt.arch_interaction_itself:
                num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
            else:
                num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
        elif self.opt.arch_interaction_op == "cat":
            num_int = num_fea * m_den_out
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.opt.arch_interaction_op
                + " is not supported"
            )
        arch_mlp_top_adjusted = str(num_int) + "-" + self.opt.arch_mlp_top
        self.opt.ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

        dlrm = DLRM_Net(
            self.opt.m_spa,
            self.opt.ln_emb,
            self.opt.ln_bot,
            self.opt.ln_top,
            arch_interaction_op=self.opt.arch_interaction_op,
            arch_interaction_itself=self.opt.arch_interaction_itself,
            sigmoid_bot=self.opt.sigmoid_bot,
            sigmoid_top=self.opt.sigmoid_top,
            sync_dense_params=self.opt.sync_dense_params,
            loss_threshold=self.opt.loss_threshold,
            ndevices=self.opt.ndevices,
            qr_flag=self.opt.qr_flag,
            qr_operation=self.opt.qr_operation,
            qr_collisions=self.opt.qr_collisions,
            qr_threshold=self.opt.qr_threshold,
            md_flag=self.opt.md_flag,
            md_threshold=self.opt.md_threshold,
        )

        # Preparing data
        for j, (X, lS_o, lS_i, T) in enumerate(self.train_ld):
            if self.device == "cuda":
                lS_i = [S_i.to(self.device) for S_i in lS_i] if isinstance(lS_i, list) \
                    else lS_i.to(self.device)
                lS_o = [S_o.to(self.device) for S_o in lS_o] if isinstance(lS_o, list) \
                    else lS_o.to(self.device)
                train_ld[j] = (X.to(self.device), lS_o, lS_i, T.to(self.device))

        # Setting Loss Function
        if self.opt.loss_function == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        elif self.opt.loss_function == "bce":
            self.loss_fn = torch.nn.BCELoss(reduction="mean")
        elif self.opt.loss_function == "wbce":
            self.loss_ws = torch.tensor(np.fromstring(self.opt.loss_weights, dtype=float, sep="-"))
            self.loss_fn = torch.nn.BCELoss(reduction="none")
        else:
            sys.exit("ERROR: --loss-function=" + self.opt.loss_function + " is not supported")

        self.module = dlrm.to(self.device)
        self.example_inputs = self.train_ld
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.SGD(dlrm.parameters(), lr=self.opt.learning_rate)
        self.lr_scheduler = LRPolicyScheduler(self.optimizer, self.opt.lr_num_warmup_steps, self.opt.lr_decay_start_step,
                                            self.opt.lr_num_decay_steps)

    def get_module(self):
        return self.module, self.example_inputs

    def eval(self, X, lS_o, lS_i, niter=1):
        self.module.eval()
        for _ in range(niter):
            self.module(X,lS_o,lS_i)

    def train(self, Z, T, niter=1):
        self.module.train()

        for _ in range(niter):
            self.optimizer.zero_grad()
            loss = self.loss_fn(Z, T)
            if self.opt.loss_function == "wbce":
                loss_ws_ = self.loss_ws[T.data.view(-1).long()].view_as(T)
                loss = loss_ws_ * loss
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

if __name__ == '__main__':
    m = Model(device='cpu', jit=False)
    module, example_inputs = m.get_module()
    for (X, lS_o, lS_i, T) in example_inputs:
        Z = module(X,lS_o,lS_i)
        m.train(Z, T, niter=1)
        m.eval(X, lS_o, lS_i, niter=1)
