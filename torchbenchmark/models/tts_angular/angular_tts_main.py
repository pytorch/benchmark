#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import traceback
import math

import torch
import torch as T
from .model import SpeakerEncoder, AngleProtoLoss
from torch.optim.optimizer import Optimizer


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(54321)

class AttrDict(dict):
    """A custom dict which converts dict keys
    to class attributes"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):  # pylint: disable=useless-super-delegation
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                    p.data.copy_(p_data_fp32)

        return loss


CONFIG = {
    "run_name": "mueller91",
    "run_description": "train speaker encoder with voxceleb1, voxceleb2 and libriSpeech ",
    "audio":{
        # Audio processing parameters
        "num_mels": 40,         # size of the mel spec frame.
        "fft_size": 400,       # number of stft frequency levels. Size of the linear spectogram frame.
        "sample_rate": 16000,   # DATASET-RELATED: wav sample-rate. If different than the original data, it is resampled.
        "win_length": 400,     # stft window length in ms.
        "hop_length": 160,      # stft window hop-lengh in ms.
        "frame_length_ms": None,  # stft window length in ms.If None, 'win_length' is used.
        "frame_shift_ms": None,   # stft window hop-lengh in ms. If None, 'hop_length' is used.
        "preemphasis": 0.98,    # pre-emphasis to reduce spec noise and make it more structured. If 0.0, no -pre-emphasis.
        "min_level_db": -100,   # normalization range
        "ref_level_db": 20,     # reference level db, theoretically 20db is the sound of air.
        "power": 1.5,           # value to sharpen wav signals after GL algorithm.
        "griffin_lim_iters": 60,# #griffin-lim iterations. 30-60 is a good range. Larger the value, slower the generation.
        # Normalization parameters
        "signal_norm": True,    # normalize the spec values in range [0, 1]
        "symmetric_norm": True, # move normalization to range [-1, 1]
        "max_norm": 4.0,          # scale normalization to range [-max_norm, max_norm] or [0, max_norm]
        "clip_norm": True,      # clip normalized values into the range.
        "mel_fmin": 0.0,         # minimum freq level for mel-spec. ~50 for male and ~95 for female voices. Tune for dataset!!
        "mel_fmax": 8000.0,        # maximum freq level for mel-spec. Tune for dataset!!
        "do_trim_silence": True,  # enable trimming of slience of audio as you load it. LJspeech (False), TWEB (False), Nancy (True)
        "trim_db": 60          # threshold for timming silence. Set this according to your dataset.
    },
    "reinit_layers": [],
    "loss": "angleproto", # "ge2e" to use Generalized End-to-End loss and "angleproto" to use Angular Prototypical loss (new SOTA)
    "grad_clip": 3.0, # upper limit for gradients for clipping.
    "epochs": 1000, # total number of epochs to train.
    "lr": 0.0001, # Initial learning rate. If Noam decay is active, maximum learning rate.
    "lr_decay": False, # if True, Noam learning rate decaying is applied through training.
    "warmup_steps": 4000, # Noam decay steps to increase the learning rate from 0 to "lr"
    "tb_model_param_stats": False, # True, plots param stats per layer on tensorboard. Might be memory consuming, but good for debugging.
    "steps_plot_stats": 10, # number of steps to plot embeddings.
    "num_speakers_in_batch": 64, # Batch size for training. Lower values than 32 might cause hard to learn attention. It is overwritten by 'gradual_training'.
    "num_utters_per_speaker": 10,  #
    "num_loader_workers": 8,        # number of training data loader processes. Don't set it too big. 4-8 are good values.
    "wd": 0.000001, # Weight decay weight.
    "checkpoint": True, # If True, it saves checkpoints per "save_step"
    "save_step": 1000, # Number of training steps expected to save traning stats and checkpoints.
    "print_step": 20, # Number of steps to log traning on console.
    "output_path": "../../MozillaTTSOutput/checkpoints/voxceleb_librispeech/speaker_encoder/", # DATASET-RELATED: output path for all training outputs.
    "model": {
        "input_dim": 40,
        "proj_dim": 256,
        "lstm_dim": 768,
        "num_lstm_layers": 3,
        "use_lstm_with_projection": True
    },
    "storage": {
        "sample_from_storage_p": 0.66,
        "storage_size": 15,   # the size of the in-memory storage with respect to a single batch
        "additive_noise": 1e-5   # add very small gaussian noise to the data in order to increase robustness
    },
    "datasets":
        [
            {
                "name": "vctk_slim",
                "path": "../../../audio-datasets/en/VCTK-Corpus/",
                "meta_file_train": None,
                "meta_file_val": None
            },
            {
                "name": "libri_tts",
                "path": "../../../audio-datasets/en/LibriTTS/train-clean-100",
                "meta_file_train": None,
                "meta_file_val": None
            },
            {
                "name": "libri_tts",
                "path": "../../../audio-datasets/en/LibriTTS/train-clean-360",
                "meta_file_train": None,
                "meta_file_val": None
            },
            {
                "name": "libri_tts",
                "path": "../../../audio-datasets/en/LibriTTS/train-other-500",
                "meta_file_train": None,
                "meta_file_val": None
            },
            {
                "name": "voxceleb1",
                "path": "../../../audio-datasets/en/voxceleb1/",
                "meta_file_train": None,
                "meta_file_val": None
            },
            {
                "name": "voxceleb2",
                "path": "../../../audio-datasets/en/voxceleb2/",
                "meta_file_train": None,
                "meta_file_val": None
            },
            {
                "name": "common_voice",
                "path": "../../../audio-datasets/en/MozillaCommonVoice",
                "meta_file_train": "train.tsv",
                "meta_file_val": "test.tsv"
            }
        ]
}


class TTSModel:
    def __init__(self, device, batch_size):
        self.device = device
        self.use_cuda = True if self.device == 'cuda' else False

        self.SYNTHETIC_DATA = []
        self.c = AttrDict()
        self.c.update(CONFIG)
        c = self.c
        self.model = SpeakerEncoder(input_dim=c.model['input_dim'],
                                    proj_dim=c.model['proj_dim'],
                                    lstm_dim=c.model['lstm_dim'],
                                    num_lstm_layers=c.model['num_lstm_layers'])
        self.optimizer = RAdam(self.model.parameters(), lr=c.lr)
        self.criterion = AngleProtoLoss()
        self.SYNTHETIC_DATA.append(T.rand(batch_size, 50, 40).to(device=self.device))

        if self.use_cuda:
            self.model = self.model.cuda()
            self.criterion.cuda()

        self.scheduler = None
        self.global_step = 0

    def __del__(self):
        del self.SYNTHETIC_DATA[0]

    def train(self):
        niter = 1
        _, global_step = self._train(self.model, self.criterion,
                                     self.optimizer, self.scheduler, None,
                                     self.global_step, self.c, niter)

    def eval(self):
        result = self.model(self.SYNTHETIC_DATA[0])
        return result

    def __call__(self, *things):
        return self

    def _train(self, model, criterion, optimizer, scheduler, ap, global_step, c, niter):
        # data_loader = setup_loader(ap, is_val=False, verbose=True)
        model.train()
        epoch_time = 0
        best_loss = float('inf')
        avg_loss = 0
        avg_loader_time = 0
        end_time = time.time()
        # for _, data in enumerate(data_loader):
        start_time = time.time()
        for reps in range(niter):
            for _, data in enumerate(self.SYNTHETIC_DATA):
                # setup input data
                # inputs = data[0]
                inputs = data
                loader_time = time.time() - end_time
                global_step += 1

                # setup lr
                # if c.lr_decay:
                #     scheduler.step()
                optimizer.zero_grad()

                # dispatch data to GPU
                if self.use_cuda:
                    inputs = inputs.cuda(non_blocking=True)
                    # labels = labels.cuda(non_blocking=True)

                # forward pass model
                outputs = model(inputs)

                # print(outputs.shape)
                view = outputs.view(c.num_speakers_in_batch, outputs.shape[0] // c.num_speakers_in_batch, -1)
                # loss computation
                loss = criterion(view)
                loss.backward()
                # grad_norm, _ = check_update(model, c.grad_clip)
                optimizer.step()

                step_time = time.time() - start_time
                epoch_time += step_time

                # Averaged Loss and Averaged Loader Time
                avg_loss = 0.01 * loss.item() \
                        + 0.99 * avg_loss if avg_loss != 0 else loss.item()
                avg_loader_time = 1/c.num_loader_workers * loader_time + \
                                (c.num_loader_workers-1) / c.num_loader_workers * avg_loader_time if avg_loader_time != 0 else loader_time
                current_lr = optimizer.param_groups[0]['lr']

                # save best model
                #best_loss = save_best_model(model, optimizer, avg_loss, best_loss,
                #                            OUT_PATH, global_step)

        end_time = time.time()
        # print(end_time - start_time)
        return avg_loss, global_step
