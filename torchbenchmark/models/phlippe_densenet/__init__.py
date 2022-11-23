from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
import torch.nn as nn
import torch
from types import SimpleNamespace
import torch.utils.data as data


class DenseLayer(nn.Module):

    def __init__(self, c_in, bn_size, growth_rate, act_fn):
        """
        Inputs:
            c_in - Number of input channels
            bn_size - Bottleneck size (factor of growth rate) for the output of the 1x1 convolution. Typically between 2 and 4.
            growth_rate - Number of output channels of the 3x3 convolution
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, bn_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            act_fn(),
            nn.Conv2d(bn_size * growth_rate, growth_rate,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.cat([out, x], dim=1)
        return out


class DenseBlock(nn.Module):

    def __init__(self, c_in, num_layers, bn_size, growth_rate, act_fn):
        """
        Inputs:
            c_in - Number of input channels
            num_layers - Number of dense layers to apply in the block
            bn_size - Bottleneck size to use in the dense layers
            growth_rate - Growth rate to use in the dense layers
            act_fn - Activation function to use in the dense layers
        """
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            layers.append(
                DenseLayer(c_in=c_in + layer_idx * growth_rate,  # Input channels are original plus the feature maps from previous layers
                           bn_size=bn_size,
                           growth_rate=growth_rate,
                           act_fn=act_fn)
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


class TransitionLayer(nn.Module):

    def __init__(self, c_in, c_out, act_fn):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            # Average the output for each 2x2 pixel group
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):

    def __init__(self, num_classes=10, num_layers=[6, 6, 6, 6], bn_size=2, growth_rate=16, act_fn_name="relu", **kwargs):
        super().__init__()
        act_fn_by_name = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "gelu": nn.GELU
        }
        self.hparams = SimpleNamespace(num_classes=num_classes,
                                       num_layers=num_layers,
                                       bn_size=bn_size,
                                       growth_rate=growth_rate,
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name])
        self._create_network()
        self._init_params()

    def _create_network(self):
        # The start number of hidden channels
        c_hidden = self.hparams.growth_rate * self.hparams.bn_size

        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            # No batch norm or activation function as done inside the Dense layers
            nn.Conv2d(3, c_hidden, kernel_size=3, padding=1)
        )

        # Creating the dense blocks, eventually including transition layers
        blocks = []
        for block_idx, num_layers in enumerate(self.hparams.num_layers):
            blocks.append(
                DenseBlock(c_in=c_hidden,
                           num_layers=num_layers,
                           bn_size=self.hparams.bn_size,
                           growth_rate=self.hparams.growth_rate,
                           act_fn=self.hparams.act_fn)
            )
            # Overall output of the dense block
            c_hidden = c_hidden + num_layers * self.hparams.growth_rate
            # Don't apply transition layer on last block
            if block_idx < len(self.hparams.num_layers) - 1:
                blocks.append(
                    TransitionLayer(c_in=c_hidden,
                                    c_out=c_hidden // 2,
                                    act_fn=self.hparams.act_fn))
                c_hidden = c_hidden // 2

        self.blocks = nn.Sequential(*blocks)

        # Mapping to classification output
        self.output_net = nn.Sequential(
            # The features have not passed a non-linearity until here.
            nn.BatchNorm2d(c_hidden),
            self.hparams.act_fn(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden, self.hparams.num_classes)
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, nonlinearity=self.hparams.act_fn_name)
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
    # Source: https://github.com/phlippe/uvadlc_notebooks_benchmarking/blob/main/PyTorch/Tutorial5_Inception_ResNet_DenseNet.py
    DEFAULT_TRAIN_BSIZE = 128
    DEFAULT_EVAL_BSIZE = 128

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit,
                         batch_size=batch_size, extra_args=extra_args)
        self.model = DenseNet()
        self.model.to(device)
        self.example_inputs = (
            torch.randn((self.batch_size, 3, 32, 32), device=self.device),
        )
        self.example_target = torch.randint(0, 10, (self.batch_size,), device=self.device)
        dataset = data.TensorDataset(self.example_inputs[0], self.example_target)
        dummy_loader = data.DataLoader(dataset, batch_size=self.batch_size)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        model = self.model
        (images, ) = self.example_inputs
        model.train()
        targets = self.example_target
        output = model(images)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def eval(self):
        model = self.model
        (images, ) = self.example_inputs
        model.eval()
        with torch.no_grad():
            out = model(images)
        return (out,)
