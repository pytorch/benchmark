
import torch
import torch.optim as optim
import torchvision.models as models
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

<<<<<<< HEAD
from torchbenchmark.util.env_check import parse_extraargs
=======
from torchbenchmark.util.fx2trt import lower_to_trt

#######################################################
#
#       DO NOT MODIFY THESE FILES DIRECTLY!!!
#       USE `gen_torchvision_benchmarks.py`
#
#######################################################
>>>>>>> 863e837 (Add initial fx2trt code)

class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION

<<<<<<< HEAD
    def __init__(self, device=None, jit=False, train_bs=32, eval_bs=32, extra_args=[]):
=======
    def __init__(self, device=None, jit=False, fx2trt=False, train_bs=32):
>>>>>>> 863e837 (Add initial fx2trt code)
        super().__init__()
        self.device = device
        self.jit = jit
        self.model = models.resnet50().to(self.device)
        self.eval_model = models.resnet50().to(self.device)
<<<<<<< HEAD
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
=======
        self.batch_size = train_bs
        # Turn on fp16 for inference by default
        self.eval_fp16 = True
        self.example_inputs = (torch.randn((self.batch_size, 3, 224, 224)).to(self.device),)

        if self.fx2trt:
            assert not self.jit, "fx2trt with JIT is not available."
            self.eval_model = lower_to_trt(max_batch_size=self.batch_size, fp16_mode=self.eval_fp16)
>>>>>>> 863e837 (Add initial fx2trt code)

        if self.jit:
            if hasattr(torch.jit, '_script_pdt'):
                self.model = torch.jit._script_pdt(self.model, example_inputs=[self.example_inputs, ])
                self.eval_model = torch.jit._script_pdt(self.eval_model)
            else:
                self.model = torch.jit.script(self.model, example_inputs=[self.example_inputs, ])
                self.eval_model = torch.jit.script(self.eval_model)
            # model needs to in `eval`
            # in order to be optimized for inference
            self.eval_model.eval()
            self.eval_model = torch.jit.optimize_for_inference(self.eval_model)


    def get_module(self):
        return self.model, self.example_inputs

    # vision models have another model
    # instance for inference that has
    # already been optimized for inference
    def set_eval(self):
        pass

    def train(self, niter=3):
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            optimizer.zero_grad()
            pred = self.model(*self.example_inputs)
            y = torch.empty(pred.shape[0], dtype=torch.long, device=self.device).random_(pred.shape[1])
            loss(pred, y).backward()
            optimizer.step()

    def eval(self, niter=1):
        model = self.eval_model
        example_inputs = self.eval_example_inputs
        for i in range(niter):
            model(*example_inputs)
