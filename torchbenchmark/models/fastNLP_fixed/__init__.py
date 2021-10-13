"""
fastNLP model (TorchBenchmark Version)
This model resembles the "BertEmedding Q&A" task in [fastNLP Tutorial](https://fastnlp.readthedocs.io/zh/latest/tutorials/extend_1_bert_embedding.html).

Input data simulates [CMRC2018 dataset](https://ymcui.com/cmrc2018/).
The program runs only for benchmark purposes and doesn't provide correctness results.
"""
import torch
import random
import numpy as np
from fastNLP.models import BertForQuestionAnswering
from fastNLP.core.losses import CMRC2018Loss
from fastNLP.core.metrics import CMRC2018Metric

# Import CMRC2018 data generator
from .cmrc2018_simulator import generate_dev, generate_train, CMRC2018_DIR

# TorchBench imports
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Model(BenchmarkModel):
    task = NLP.OTHER_NLP
    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit
        self.input_dir = CMRC2018_DIR
        generate_dev()
        generate_train()
        # # TODO: embed (preprocessing result)
        # self.model = _move_model_to_device(BertForQuestionAnswering(embed), device=device)
        # if _model_contains_inner_module(self.model):
        #     self._forward_func = self.model.module.forward
        # else:
        #     self._forward_func = self.model.forward
        # self.losser = CMRC2018Loss()
        # self.metrics = CMRC2018Metric()
        # # TODO: get the default batchsize
        # self.batch_size = 6
        # wm_callback = WarmupCallback(schedule='linear')
        # gc_callback = GradientClipCallback(clip_value=1, clip_type='norm')
        # callbacks = [wm_callback, gc_callback]
        # self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        # self.callback_manager = CallbackManager(env={"trainer":self}, callbacks=callbacks)

    # Run with example input
    def get_module(self):
        pass

    # Sliced version of fastNLP.Predictor._predict()
    def eval(self, niter=1):
        pass

    # Sliced version of fastNLP.Trainer._train()
    def train(self, niter=1):
        pass
        # Loss begin
        # Loss end
        # Callback_manager
        # 
        # callback_manager.on_epoch_end()

    # Helper functions
    def _mode(self, model, is_test=False):
        r"""Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param bool is_test: whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()

    def _update(self):
        r"""Perform weight update on a model.
        """
        if self.step % self.update_every == 0:
            self.optimizer.step()

    def _data_forward(self, network, x):
        x = _build_args(self._forward_func, **x)
        y = network(**x)
        if not isinstance(y, dict):
            raise TypeError(
                f"The return value of {_get_func_signature(self._forward_func)} should be dict, got {type(y)}.")
        return y

    def _grad_backward(self, loss):
        r"""Compute gradient with link rules.

        :param loss: a scalar where back-prop starts

        For PyTorch, just do "loss.backward()"
        """
        if (self.step-1) % self.update_every == 0:
            self.model.zero_grad()
        loss.backward()

    def _compute_loss(self, predict, truth):
        r"""Compute loss given prediction and ground truth.

        :param predict: prediction dict, produced by model.forward
        :param truth: ground truth dict, produced by batch_y
        :return: a scalar
        """
        return self.losser(predict, truth)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = Model(device=device, jit=False)
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()
