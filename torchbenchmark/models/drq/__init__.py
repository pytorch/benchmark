import copy
import math
import pickle as pkl
import numpy as np
import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import dmc2gym

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import REINFORCEMENT_LEARNING

from .utils import FrameStack, set_seed_everywhere, eval_mode
from .drq import DRQAgent
from .config import DRQConfig
from .replay_buffer import ReplayBuffer

def make_env(cfg):
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])
     # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'quadruped' else 0

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=cfg.image_size,
                       width=cfg.image_size,
                       frame_skip=cfg.action_repeat,
                       camera_id=camera_id)

    env = FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

class Model(BenchmarkModel):
    task = REINFORCEMENT_LEARNING.OTHER_RL
    def __init__(self, device=None, jit=False):
        super(Model, self).__init__()
        self.device = device
        self.jit = jit
        self.cfg = DRQConfig()
        set_seed_everywhere(self.cfg.seed)
        self.env = make_env(self.cfg)
        obs_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = DRQAgent(self.cfg, self.device, obs_shape, action_shape, action_range)
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          self.cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device)
        self.step = 0

    def get_module(self):
        pass

    def train(self, niter=1):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        for step in range(niter):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
            if step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger, self.step)
            next_obs, reward, done, info = self.env.step(action)
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward
            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)
            obs = next_obs
            episode_step += 1
            self.step += 1

    def eval(self, niter=1):
        average_episode_reward = 0
        for episode in range(niter):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_step += 1
            average_episode_reward += episode_reward
        average_episode_reward /= float(niter)
