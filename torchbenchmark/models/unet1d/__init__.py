from ...util.model import BenchmarkModel

import torch
from diffusers import UNet1DModel

class Model(BenchmarkModel):
    DEFAULT_EVAL_BSIZE = 32
    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__( test=test, device=device,
                batch_size=batch_size, extra_args=extra_args)

        self.in_channels = 32
        self.seq_len = 256
        self.num_features = 16
        self.block_out_channels = (self.seq_len, 64, 64)
        print(self.batch_size)
        self.example_inputs =  torch.randn(1, self.num_features, self.seq_len)
        self.timesteps = torch.tensor([1])
        self.model = UNet1DModel(in_channels=self.in_channels, block_out_channels=self.block_out_channels)

    def get_module(self):
        return self.model, self.example_inputs

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            out=self.model(self.example_inputs, self.timesteps)
