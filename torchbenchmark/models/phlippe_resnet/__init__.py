"""
https://github.com/phlippe/uvadlc_notebooks_benchmarking/blob/main/PyTorch/Tutorial5_Inception_ResNet_DenseNet.py
"""
from types import SimpleNamespace
from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.model import BenchmarkModel
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as data


class ResNetBlock(nn.Module):

    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in
        # Network representing F
        self.net = nn.Sequential(
            # No bias needed as the Batch Norm handles it
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1,
                      stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv2d(
            c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out


class ResNetModel(nn.Module):

    def __init__(self, num_classes=10, num_blocks=[3, 3, 3], c_hidden=[16, 32, 64], act_fn_name="relu", **kwargs):
        """
        Inputs:
            num_classes - Number of classification outputs (10 for CIFAR10)
            num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
            act_fn_name - Name of the activation function to use, looked up in "act_fn_by_name"
            block_name - Name of the ResNet block, looked up in "resnet_blocks_by_name"
        """
        super().__init__()
        act_fn_by_name = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "gelu": nn.GELU
        }
        self.hparams = SimpleNamespace(num_classes=num_classes,
                                       c_hidden=c_hidden,
                                       num_blocks=num_blocks,
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name],
                                       block_class=ResNetBlock)
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden
        self.input_net = nn.Sequential(
            nn.Conv2d(3, c_hidden[0], kernel_size=3,
                        padding=1, bias=False),
            nn.BatchNorm2d(c_hidden[0]),
            self.hparams.act_fn()
        )
        # Creating the ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                blocks.append(
                    self.hparams.block_class(c_in=c_hidden[block_idx if not subsample else (block_idx - 1)],
                                             act_fn=self.hparams.act_fn,
                                             subsample=subsample,
                                             c_out=c_hidden[block_idx])
                )
        self.blocks = nn.Sequential(*blocks)

        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.hparams.num_classes)
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x


class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION
    DEFAULT_TRAIN_BSIZE = 128
    DEFAULT_EVAL_BSIZE = 128

    def __init__(self, test, device, jit=False, batch_size=DEFAULT_TRAIN_BSIZE, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit,
                         batch_size=batch_size, extra_args=extra_args)
        self.model = ResNetModel()
        self.model.to(device)
        self.example_inputs = (
            torch.randn((self.batch_size, 3, 32, 32), device=self.device),
        )
        self.example_target = torch.randint(0, 10, (self.batch_size,), device=self.device)
        dataset = data.TensorDataset(self.example_inputs[0], self.example_target)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        (self.images, ) = self.example_inputs

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        self.model.train()
        targets = self.example_target
        output = self.model(self.images)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def eval(self):
        self.model.eval()
        with torch.no_grad():
            out=self.model(self.images)
        return (out,)
