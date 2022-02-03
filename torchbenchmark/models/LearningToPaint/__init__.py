import cv2
import torch
import random
import numpy as np
from .baseline.Renderer.model import FCN
from .baseline.DRL.evaluator import Evaluator
from .baseline.utils.util import *
from .baseline.DRL.ddpg import DDPG
from .baseline.DRL.multi import fastenv

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import REINFORCEMENT_LEARNING

from argparse import Namespace

torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


class Model(BenchmarkModel):
    task = REINFORCEMENT_LEARNING.OTHER_RL

    def __init__(self, test="eval", device=None, jit=False, train_bs=96, eval_bs=96):
        super(Model, self).__init__()
        self.device = device
        self.jit = jit
        # Train: These options are from source code.
        # Source: https://arxiv.org/pdf/1903.04411.pdf
        # Code: https://github.com/megvii-research/ICCV2019-LearningToPaint/blob/master/baseline/train.py
        self.args = Namespace(**{
            'validate_episodes': 5,
            'validate_interval': 50,
            'max_step': 40,
            'discount': 0.95**5,
            'episode_train_times': 10,
            'noise_factor': 0.0,
            'tau': 0.001,
            'rmsize': 800,
        })
        self.train_bs = train_bs
        self.eval_bs = eval_bs

        # Train: input images are from CelebFaces and resized to 128 x 128.
        # Create 2000 random tensors for input, but randomly sample 200,000 images.
        self.width = 128
        self.image_examples = torch.rand(2000, 3, self.width, self.width)

        # LearningToPaint includes actor, critic, and discriminator models.
        self.Decoder = FCN()
        self.train_env = fastenv(max_episode_length=self.args.max_step, env_batch=self.train_bs,
                                 images=self.image_examples, device=self.device, Decoder=self.Decoder)
        self.train_agent = DDPG(batch_size=self.train_bs, env_batch=self.train_bs,
                                max_step=self.args.max_step, tau=self.args.tau, discount=self.args.discount,
                                rmsize=self.args.rmsize, device=self.device, Decoder=self.Decoder)
        self.train_evaluate = Evaluator(args=self.args, env_batch=train_bs, writer=None)
        self.train_agent.train()

        self.infer_env = fastenv(max_episode_length=self.args.max_step, env_batch=self.eval_bs,
                                 images=self.image_examples, device=self.device, Decoder=self.Decoder)
        self.infer_agent = DDPG(batch_size=self.eval_bs, env_batch=self.eval_bs,
                                max_step=self.args.max_step, tau=self.args.tau, discount=self.args.discount,
                                rmsize=self.args.rmsize, device=self.device, Decoder=self.Decoder)
        self.infer_evaluate = Evaluator(args=self.args, env_batch=eval_bs, writer=None)
        self.infer_agent.eval()

        self.step = 0
        self.train_observation = self.train_env.reset()
        self.infer_observation = self.infer_env.reset()
        self.train_agent.reset(self.train_observation, self.args.noise_factor)
        self.infer_agent.reset(self.infer_observation, self.args.noise_factor)

    def get_module(self):
        action = self.train_agent.select_action(self.train_observation, noise_factor=self.args.noise_factor)
        self.train_observation, reward, done, _ = self.train_env.step(action)
        self.train_agent.observe(reward, self.train_observation, done, self.step)
        state, action, reward, \
            next_state, terminal = self.train_agent.memory.sample_batch(self.train_bs, self.device)
        state = torch.cat((state[:, :6].float() / 255, state[:, 6:7].float() / self.args.max_step,
                           self.train_agent.coord.expand(state.shape[0], 2, 128, 128)), 1)
        return self.train_agent.actor, (state, )

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        episode = episode_steps = 0
        for _ in range(niter):
            episode_steps += 1
            if self.train_observation is None:
                self.train_observation = self.train_env.reset()
                self.train_agent.reset(self.train_observation, self.args.noise_factor)
            action = self.train_agent.select_action(self.train_observation, noise_factor=self.args.noise_factor)
            self.train_observation, reward, done, _ = self.train_env.step(action)
            self.train_agent.observe(reward, self.train_observation, done, self.step)
            if (episode_steps >= self.args.max_step and self.args.max_step):
                # [optional] evaluate
                if episode > 0 and self.args.validate_interval > 0 and \
                        episode % self.args.validate_interval == 0:
                    reward, dist = self.train_evaluate(self.train_env, self.train_agent.select_action)
                tot_Q = 0.
                tot_value_loss = 0.
                lr = (3e-4, 1e-3)
                for i in range(self.args.episode_train_times):
                    Q, value_loss = self.train_agent.update_policy(lr)
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()
                # reset
                self.train_observation = None
                episode_steps = 0
                episode += 1
            self.step += 1

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        for _ in range(niter):
            reward, dist = self.infer_evaluate(self.infer_env, self.infer_agent.select_action)

    # Using separate models for train and infer, so skip this function.
    def _set_mode(self, train):
        pass

if __name__ == '__main__':
    m = Model(device='cpu', jit=False)
    module, example_inputs = m.get_module()
    while m.step < 100:
        m.train(niter=1)
        if m.step % 100 == 0:
            m.eval(niter=1)
        m.step += 1
