import time
import random
import torch

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER
from typing import Tuple

from pplbench.ppls.beanmachine import robust_regression
from pplbench.models.robust_regression import RobustRegression
from pplbench.ppls.beanmachine.inference import MCMC

class Pplbench(torch.nn.Module):
    def __init__(self, test):
        super(Pplbench, self).__init__()

        # Instantiate model
        self.model = RobustRegression()

        # Get data for inference & evaluating model
        if test == "eval":
            # Increased number of samples (n) and number of features(k) to increase computation for eval
            # Default values Reference: https://github.com/facebookresearch/pplbench/blob/main/examples/robust_regression.json
            self.train_data, self.test_data = self.model.generate_data(seed=int(time.time()), n=500000, k=500)
        else:
            self.train_data, self.test_data = self.model.generate_data(seed=int(time.time()))

        # Create inference object with training data
        self.infer_obj = MCMC(robust_regression.RobustRegression,
                              self.train_data.attrs)
        self.infer_obj.compile()

        if test == "eval":
            # Run bayesian inference (training) on given data for 1 iteration
            # We need the object of type MonteCarloSamples for the evaluation step
            # @Todo Can we create samples object without using infer function?
            self.samples = self.infer_obj.infer(data=self.train_data, iterations=1, num_warmup=0, seed=random.randint(1, int(1e7)))

    def forward(self, train_data, test_data, training=False):

        if training:
            # Run bayesian inference (training) on given data
            # Reference: https://github.com/facebookresearch/pplbench/blob/main/examples/robust_regression.json
            samples = self.infer_obj.infer(data=train_data, iterations=1500, num_warmup=0, seed=random.randint(1, int(1e7)))
        else:
            # Evaluate the model with test data and compute the posterior probabilities
            out = self.model.evaluate_posterior_predictive(self.samples, test_data)
            return torch.Tensor(out)


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS

    # Batch size is not adjustable in the model
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        if device != "cpu":
            raise NotImplementedError("The {} test only supports CPU.".format(test))

        # Instantiate model
        self.model = Pplbench(test)

        self.example_inputs = (self.model.train_data, self.model.test_data)

    def get_module(self):
        return self.model, self.example_inputs

    def train(self, niter=1):
        model, example_inputs = self.get_module()

        _ = model(*example_inputs, training=True)

    def eval(self, niter=1) -> Tuple[torch.Tensor]:
        model, example_inputs = self.get_module()

        out = model(*example_inputs)
        return (out, )
