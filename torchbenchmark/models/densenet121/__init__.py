import torch
import torch.optim as optim
import torchvision.models as models
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from torchbenchmark.util.env_check import parse_extraargs

class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Train batch size: use the training batch in paper.
    # Source: https://arxiv.org/pdf/1608.06993.pdf
    def __init__(self, device=None, jit=False, train_bs=256, eval_bs=64, extra_args=[]):
        super().__init__()
        self.device = device
        self.jit = jit
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
        self.extra_args = parse_extraargs(extra_args)
        if self.extra_args.eval_fp16:
            self.eval_model.half()
            self.eval_example_inputs = (self.eval_example_inputs[0].half(),)
        if self.extra_args.fx2trt:
            assert self.device == 'cuda', "fx2trt is only available with CUDA."
            assert not self.jit, "fx2trt with JIT is not available."
            from torchbenchmark.util.fx2trt import lower_to_trt
            self.eval_model = lower_to_trt(module=self.eval_model, input=self.eval_example_inputs, \
                                           max_batch_size=eval_bs, fp16_mode=self.extra_args.eval_fp16)

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
