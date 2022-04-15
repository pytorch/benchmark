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
    def __init__(self):
        super(Pplbench, self).__init__()

        # Instantiate model
        self.model = RobustRegression()

        # Get data for evaluating model
        self.train_data, self.test_data = self.model.generate_data(seed=int(time.time()))

        # Create inference object with training data
        self.infer_obj = MCMC(robust_regression.RobustRegression,
                              self.train_data.attrs)
        self.infer_obj.compile()

    def forward(self, train_data, test_data):

        # Run inference on given data
        samples = self.infer_obj.infer(data=train_data, iterations=1, num_warmup=0, seed=random.randint(1, int(1e7)))

        # Evaluate the model with test data
        out = self.model.evaluate_posterior_predictive(samples, test_data)

        return torch.Tensor(out)

class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        if test == "eval" and device != "cpu":
            raise NotImplementedError("The eval test only supports CPU.")

        # Instantiate model
        self.model = Pplbench()

        self.example_inputs = (self.model.train_data, self.model.test_data)

    def get_module(self):
        return self.model, self.example_inputs

    def train(self, niter=1):
        raise NotImplementedError("Training not supported")

    def eval(self, niter=1) -> Tuple[torch.Tensor]:
        model, example_inputs = self.get_module()

        out = model(*example_inputs)
        return (out, )
