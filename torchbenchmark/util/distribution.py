import math

import torch
import torch.nn.functional as F
from torch import distributions as pyd


"""
Credit for actor distribution code: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
"""


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1, clamp=None):
        super().__init__(cache_size=cache_size)
        self.clamp = clamp

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y.clamp(self.clamp))

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale, tanh_transform_clamp=None):
        self.loc = loc
        self.scale = scale
        self.tanh_transform_clamp = tanh_transform_clamp
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform(clamp=tanh_transform_clamp)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def _squashed_normal_flatten(t: SquashedNormal):
    return [t.loc, t.scale], t.tanh_transform_clamp


def _squashed_normal_unflatten(values, context):
    return SquashedNormal(*values, context)


torch.utils._pytree.register_pytree_node(
    SquashedNormal,
    _squashed_normal_flatten,
    _squashed_normal_unflatten,
    serialized_type_name=f"{SquashedNormal.__module__}.{SquashedNormal.__name__}",
)
