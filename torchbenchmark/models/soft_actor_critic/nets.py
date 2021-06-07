import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn

from . import utils


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class BigPixelEncoder(nn.Module):
    def __init__(self, obs_shape, out_dim=50):
        super().__init__()
        channels = obs_shape[0]
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        output_height, output_width = utils.compute_conv_output(
            obs_shape[1:], kernel_size=(3, 3), stride=(2, 2)
        )
        for _ in range(3):
            output_height, output_width = utils.compute_conv_output(
                (output_height, output_width), kernel_size=(3, 3), stride=(1, 1)
            )

        self.fc = nn.Linear(output_height * output_width * 32, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.apply(weight_init)

    def forward(self, obs):
        obs /= 255.0
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.ln(x)
        state = torch.tanh(x)
        return state


class SmallPixelEncoder(nn.Module):
    def __init__(self, obs_shape, out_dim=50):
        super().__init__()
        channels = obs_shape[0]
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        output_height, output_width = utils.compute_conv_output(
            obs_shape[1:], kernel_size=(8, 8), stride=(4, 4)
        )

        output_height, output_width = utils.compute_conv_output(
            (output_height, output_width), kernel_size=(4, 4), stride=(2, 2)
        )

        output_height, output_width = utils.compute_conv_output(
            (output_height, output_width), kernel_size=(3, 3), stride=(1, 1)
        )

        self.fc = nn.Linear(output_height * output_width * 64, out_dim)
        self.apply(weight_init)

    def forward(self, obs):
        obs /= 255.0
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        state = self.fc(x)
        return state


class StochasticActor(nn.Module):
    def __init__(
        self,
        state_space_size,
        act_space_size,
        log_std_low=-10,
        log_std_high=2,
        hidden_size=1024,
        dist_impl="pyd",
    ):
        super().__init__()
        assert dist_impl in ["pyd", "beta"]
        self.fc1 = nn.Linear(state_space_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2 * act_space_size)
        self.log_std_low = log_std_low
        self.log_std_high = log_std_high
        self.apply(weight_init)
        self.dist_impl = dist_impl

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        mu, log_std = out.chunk(2, dim=1)
        if self.dist_impl == "pyd":
            log_std = torch.tanh(log_std)
            log_std = self.log_std_low + 0.5 * (
                self.log_std_high - self.log_std_low
            ) * (log_std + 1)
            std = log_std.exp()
            dist = SquashedNormal(mu, std)
        elif self.dist_impl == "beta":
            out = 1.0 + F.softplus(out)
            alpha, beta = out.chunk(2, dim=1)
            dist = BetaDist(alpha, beta)
        return dist


class BigCritic(nn.Module):
    def __init__(self, state_space_size, act_space_size, hidden_size=1024):
        super().__init__()
        self.fc1 = nn.Linear(state_space_size + act_space_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.apply(weight_init)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat((state, action), dim=1)))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class BaselineActor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=400):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        act = torch.tanh(self.out(x))
        return act


class BaselineCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        val = self.out(x)
        return val


class BetaDist(pyd.transformed_distribution.TransformedDistribution):
    class _BetaDistTransform(pyd.transforms.Transform):
        domain = pyd.constraints.real
        codomain = pyd.constraints.interval(-1.0, 1.0)

        def __init__(self, cache_size=1):
            super().__init__(cache_size=cache_size)

        def __eq__(self, other):
            return isinstance(other, _BetaDistTransform)

        def _inverse(self, y):
            return (y.clamp(-0.99, 0.99) + 1.0) / 2.0

        def _call(self, x):
            return (2.0 * x) - 1.0

        def log_abs_det_jacobian(self, x, y):
            # return log det jacobian |dy/dx| given input and output
            return torch.Tensor([math.log(2.0)]).to(x.device)

    def __init__(self, alpha, beta):
        self.base_dist = pyd.beta.Beta(alpha, beta)
        transforms = [self._BetaDistTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.base_dist.mean
        for tr in self.transforms:
            mu = tr(mu)
        return mu


"""
Credit for actor distribution code: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
"""


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y.clamp(-0.99, 0.99))

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class GracBaselineActor(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_mean = nn.Linear(300, action_size)
        self.fc_std = nn.Linear(300, action_size)

    def forward(self, state, stochastic=False):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc_mean(x))
        std = F.softplus(self.fc_std(x)) + 1e-3
        dist = pyd.Normal(mean, std)
        return dist


class BaselineDiscreteActor(nn.Module):
    def __init__(self, obs_shape, action_size, hidden_size=300):
        super().__init__()
        self.fc1 = nn.Linear(obs_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act_p = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        act_p = F.softmax(self.act_p(x), dim=1)
        dist = pyd.categorical.Categorical(act_p)
        return dist


class BaselineDiscreteCritic(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size=300):
        super().__init__()
        self.fc1 = nn.Linear(obs_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_shape)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        vals = self.out(x)
        return vals
