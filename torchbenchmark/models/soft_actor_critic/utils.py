import math
import os
import random
from collections import namedtuple

import gym
import numpy as np
import torch


def clean_hparams_dict(hparams_dict):
    return {key: val for key, val in hparams_dict.items() if val}


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        try:
            param = p.grad.data
        except AttributeError:
            continue
        else:
            param_norm = param.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def torch_and_pad(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)


def mean(lst):
    return float(sum(lst)) / len(lst)


def make_process_dirs(run_name, base_path="dc_saves"):
    base_dir = os.path.join(base_path, run_name)
    i = 0
    while os.path.exists(base_dir + f"_{i}"):
        i += 1
    base_dir += f"_{i}"
    os.makedirs(base_dir)
    return base_dir


def compute_conv_output(
    inp_shape, kernel_size, padding=(0, 0), dilation=(1, 1), stride=(1, 1)
):
    """
    Compute the shape of the output of a torch Conv2d layer using 
    the formula from the docs.

    every argument is a tuple corresponding to (height, width), e.g. kernel_size=(3, 4)
    """
    height_out = math.floor(
        (
            (inp_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
            / stride[0]
        )
        + 1
    )
    width_out = math.floor(
        (
            (inp_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
            / stride[1]
        )
        + 1
    )
    return height_out, width_out


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


""" This is all from: https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py """


class AnnealedGaussianProcess:
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.0
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(
        self,
        theta,
        mu=0.0,
        sigma=1.0,
        dt=1e-2,
        x0=None,
        size=1,
        sigma_min=None,
        n_steps_annealing=1000,
    ):
        super(OrnsteinUhlenbeckProcess, self).__init__(
            mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing
        )
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class GaussianExplorationNoise:
    def __init__(self, size, start_scale=1.0, final_scale=0.1, steps_annealed=1000):
        assert start_scale >= final_scale
        self.size = size
        self.start_scale = start_scale
        self.final_scale = final_scale
        self.steps_annealed = steps_annealed
        self._current_scale = start_scale
        self._scale_slope = (start_scale - final_scale) / steps_annealed

    def sample(self):
        noise = self._current_scale * torch.randn(*self.size)
        self._current_scale = max(
            self._current_scale - self._scale_slope, self.final_scale
        )
        return noise.numpy()

    def reset_states(self):
        pass
