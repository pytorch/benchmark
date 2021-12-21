import torch
import torch.optim as optim
import torchvision.models as models
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from torchbenchmark.util.framework.vision.args import parse_args, apply_args

class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION
    optimized_for_inference = True

    # Train batch size: use the training batch in paper.
    # Source: https://arxiv.org/pdf/1608.06993.pdf
    def __init__(self, device=None, jit=False, train_bs=256, eval_bs=64, extra_args=[]):
        super().__init__()
        self.device = device
        self.jit = jit
        self.train_bs = train_bs
        self.eval_bs = eval_bs

        # Temporarily disable tests because it causes CUDA OOM on CI platform
        # TODO: Re-enable these tests when better hardware is available
        if self.device == 'cuda':
            raise NotImplementedError('CUDA disabled due to CUDA out of memory on CI GPU')
        if self.device == 'cpu' and self.jit:
            raise NotImplementedError('CPU with jit disabled due to out of memory on CI CPU')
        self.model = models.densenet121().to(self.device)
        self.eval_model = models.densenet121().to(self.device)
        # Input data is ImageNet shaped as 3, 224, 224.
        self.example_inputs = (torch.randn((train_bs, 3, 224, 224)).to(self.device),)
        self.eval_example_inputs = (torch.randn((eval_bs, 3, 224, 224)).to(self.device),)

        # process extra args
        self.args = parse_args(self, extra_args)
        apply_args(self, self.args)

        if self.jit:
            if hasattr(torch.jit, '_script_pdt'):
                self.model = torch.jit._script_pdt(self.model, example_inputs=[self.example_inputs, ])
                self.eval_model = torch.jit._script_pdt(self.eval_model, example_inputs=[self.infer_example_inputs, ])
            else:
                self.model = torch.jit.script(self.model, example_inputs=[self.example_inputs, ])
                self.eval_model = torch.jit.script(self.eval_model, example_inputs=[self.infer_example_inputs, ])
            # model needs to in `eval`
            # in order to be optimized for inference
            self.eval_model.eval()
            self.eval_model = torch.jit.optimize_for_inference(self.eval_model)

    def get_module(self):
        if self.device == 'cuda':
            raise NotImplementedError('CUDA disabled due to CUDA out of memory on CI GPU')
        return self.model, self.example_inputs

    def train(self, niter=3):
        if self.device == 'cuda':
            raise NotImplementedError('CUDA disabled due to CUDA out of memory on CI GPU')
        if self.device == 'cpu':
            raise NotImplementedError('CPU disabled due to out of memory on CI CPU')
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            optimizer.zero_grad()
            pred = self.model(*self.example_inputs)
            y = torch.empty(pred.shape[0], dtype=torch.long, device=self.device).random_(pred.shape[1])
            loss(pred, y).backward()
            optimizer.step()

    def eval(self, niter=1):
        if self.device == 'cuda':
            raise NotImplementedError('CUDA disabled due to CUDA out of memory on CI GPU')
        model = self.eval_model
        example_inputs = self.eval_example_inputs
        for i in range(niter):
            model(*example_inputs)
