import copy
import math
import pickle as pkl
import numpy as np
import torch
import os
import sys
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from gym import spaces

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import REINFORCEMENT_LEARNING

from .drqutils import FrameStack, set_seed_everywhere, eval_mode
from .drq import DRQAgent
from .config import DRQConfig
from .replay_buffer import ReplayBuffer

class MockEnv:
    def __init__(self, obs):
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[1],
            dtype=np.float32)
        self._observation_space = spaces.Box(
            low=0,
            high=255,
            shape=[9, 84, 84],
            dtype=np.uint8
        )
        self.obs = obs
        self._max_episode_steps = 250
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))
    def step(self, action):
        reward = 0.0
        done = False
        info_state = [0.016243, 3.1355, -0.0052817, -0.01073]
        info = dict()
        info["internal_state"] = info_state
        info["discount"] = 1.0
        return (self.obs, reward, done, info)
    def seed(self, seed=None):
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)
    def reset(self):
        return self.obs
    @property
    def observation_space(self):
        return self._observation_space
    @property
    def action_space(self):
        return self._norm_action_space

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

    current_dir = os.path.dirname(os.path.realpath(__file__))
    mockobs = pkl.load(open(os.path.join(current_dir, cfg.obs_path), "rb"))
    low = np.amin(mockobs)
    high = np.amax(mockobs)
    mockobs = np.random.randint(low=11, high=228, size=mockobs.shape, dtype=np.uint8)
    env = MockEnv(mockobs)
    env = FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

class Model(BenchmarkModel):
    task = REINFORCEMENT_LEARNING.OTHER_RL
    # Batch size is not adjustable in this model
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False
    CANNOT_SET_CUSTOM_OPTIMIZER = True
    # this model will cause infinite loop if deep-copied
    DEEPCOPY = False

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)

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
        obs = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        return self.agent.actor, (obs, )

    def set_module(self, new_model):
        self.agent.actor = new_model

    def train(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        if True:
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, None,
                                      self.step)
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

    def eval(self) -> Tuple[torch.Tensor]:
        average_episode_reward = 0
        steps = 0
        if True:
            obs = self.env.reset()
            episode_reward = 0
            episode_step = 0
            with eval_mode(self.agent):
                action = self.agent.act(obs, sample=False)
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            episode_step += 1
            average_episode_reward += episode_reward
            steps += 1
        average_episode_reward /= float(steps)
        return (torch.Tensor(action), )
