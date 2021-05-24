import numpy as np
import torch
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import REINFORCEMENT_LEARNING

from .config import DQNConfig
from .env import Env
from .agent import Agent
from .model import DQN
from .memory import ReplayMemory

class Model(BenchmarkModel):
    task = REINFORCEMENT_LEARNING.OTHER_RL

    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit
        self.args = DQNConfig(self.device)
        if device == "cuda":
            torch.cuda.manual_seed(np.random.randint(1, 10000))
        self.env = Env(self.args)
        self.env.train()
        self.dqn = Agent(self.args, self.env)
        self.mem = ReplayMemory(self.args, self.args.memory_capacity)

    def get_module(self):
        model = self.dqn.online_net
        idxs, states, actions, returns, next_states, nonterminals, weights = self.mem.sample(self.args.batch_size)
        return model, states

    def train(self, niter = 1):
        if self.jit:
            return NotImplemented
        self.dqn.train()
        T, done = 0, True
        for T in range(1, niter + 1):
            if done:
                state = self.env.reset()
            if T % self.args.replay_frequency == 0:
                self.dqn.reset_noise()  # Draw a new set of noisy weights
            action = self.dqn.act(state)  # Choose an action greedily (with noisy weights)
            next_state, reward, done = self.env.step(action)  # Step
            if self.args.reward_clip > 0:
                reward = max(min(reward, self.args.reward_clip), -self.args.reward_clip)  # Clip rewards
            self.mem.append(state, action, reward, done)  # Append transition to memory
            if T >= self.args.learn_start:
                self.mem.priority_weight = min(self.mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
                if T % self.args.replay_frequency == 0:
                    dqn.learn(self.mem)  # Train with n-step distributional double-Q learning
            if T % self.args.target_update == 0:
                dqn.update_target_net()
            state = next_state
        self.env.close()

    def eval(self, niter = 1):
        if self.jit:
            return NotImplemented
        # Set DQN to evaluation mode
        self.dqn.eval()
        self.env.eval()
        done = True
        for iter in range(niter):
            if done:
                state, reward_sum, done = self.env.reset(), 0, False
            action = self.dqn.act_e_greedy(state)
            state, reward, done = self.env.step(action)
        self.env.close()

if __name__ == "__main__":
    m = Model(device="cuda")
    module, example_inputs = m.get_module()
    m.train(niter=1)
    m.eval(niter=1)
