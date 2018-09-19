from collections import namedtuple
import torch

from .factory import pytorch_lstm_creator, script_lstm_creator
from .factory import script_lstm_flat_inputs_creator
from .factory import script_lstm_flat_inputs_premul_creator


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
    'jit_flat': RNNRunner('jit_flat', script_lstm_flat_inputs_creator, DummyContext),
    'jit_premul': RNNRunner('jit_premul', script_lstm_flat_inputs_premul_creator, DummyContext),
    'jit': RNNRunner('jit', script_lstm_creator, DummyContext),
    'py': RNNRunner('py', script_lstm_creator, AssertNoJIT),
    'pyflat': RNNRunner('pyflat', script_lstm_flat_inputs_creator, AssertNoJIT),
}
