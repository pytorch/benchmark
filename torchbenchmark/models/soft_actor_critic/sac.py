import argparse
import copy
import math
import os
from itertools import chain

import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm

from . import envs, nets, replay, run, utils

class SACAgent:
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        log_std_low,
        log_std_high,
        device,
        actor_net_cls=nets.StochasticActor,
        critic_net_cls=nets.BigCritic,
        hidden_size=1024,
    ):
        self.actor = actor_net_cls(
            obs_space_size,
            act_space_size,
            log_std_low,
            log_std_high,
            dist_impl="pyd",
            hidden_size=hidden_size,
        )
        self.critic1 = critic_net_cls(obs_space_size, act_space_size, hidden_size)
        self.critic2 = critic_net_cls(obs_space_size, act_space_size, hidden_size)
        self.device = device

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic1 = self.critic1.to(device)
        self.critic2 = self.critic2.to(device)

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def save(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic1_path = os.path.join(path, "critic1.pt")
        critic2_path = os.path.join(path, "critic2.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)

    def load(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic1_path = os.path.join(path, "critic1.pt")
        critic2_path = os.path.join(path, "critic2.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))

    def forward(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.mean
        self.actor.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def sample_action(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.sample()
        self.actor.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(
            self.device
        )

    def process_act(self, act):
        return np.squeeze(act.clamp(-1.0, 1.0).cpu().numpy(), 0)


class SACDAgent(SACAgent):
    def __init__(self, obs_space_size, act_space_size):
        self.actor = nets.BaselineDiscreteActor(obs_space_size, act_space_size)
        self.critic1 = nets.BaselineDiscreteCritic(obs_space_size, act_space_size)
        self.critic2 = nets.BaselineDiscreteCritic(obs_space_size, act_space_size)

    def forward(self, state):
        state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = torch.argmax(act_dist.probs, dim=1)
        self.actor.train()
        return self.process_act(act)

