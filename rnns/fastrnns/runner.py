from collections import namedtuple
from functools import partial
import torch
import torchvision.models as cnn

from .factory import pytorch_lstm_creator, lstm_creator, lstm_premul_creator, \
    lstm_multilayer_creator, lstm_simple_creator, imagenet_cnn_creator


class DisableCuDNN():
    def __enter__(self):
        self.saved = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

    def __exit__(self, *args, **kwargs):
        torch.backends.cudnn.enabled = self.saved


class DummyContext():
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class AssertNoJIT():
    def __enter__(self):
        import os
        enabled = os.environ.get('PYTORCH_JIT', 1)
        assert not enabled

    def __exit__(self, *args, **kwargs):
        pass


RNNRunner = namedtuple('RNNRunner', [
    'name', 'creator', 'context',
])


def get_rnn_runners(*names):
    return [rnn_runners[name] for name in names]


rnn_runners = {
    'cudnn': RNNRunner('cudnn', pytorch_lstm_creator, DummyContext),
    'aten': RNNRunner('aten', pytorch_lstm_creator, DisableCuDNN),
    'jit': RNNRunner('jit', lstm_creator, DummyContext),
    'jit_premul': RNNRunner('jit_premul', lstm_premul_creator, DummyContext),
    'jit_simple': RNNRunner('jit_simple', lstm_simple_creator, DummyContext),
    'jit_multilayer': RNNRunner('jit_multilayer', lstm_multilayer_creator, DummyContext),
    'py': RNNRunner('py', partial(lstm_creator, script=False), DummyContext),
    'resnet18': RNNRunner('resnet18', imagenet_cnn_creator(cnn.resnet18, jit=False), DummyContext),
    'resnet18_jit': RNNRunner('resnet18_jit', imagenet_cnn_creator(cnn.resnet18), DummyContext),
    'resnet50': RNNRunner('resnet50', imagenet_cnn_creator(cnn.resnet50, jit=False), DummyContext),
    'resnet50_jit': RNNRunner('resnet50_jit', imagenet_cnn_creator(cnn.resnet50), DummyContext),
}
