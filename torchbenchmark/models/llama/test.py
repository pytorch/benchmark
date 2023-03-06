
import torch
from .model import ModelArgs, Transformer
import torch

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class Model:
    def __init__(self, temperature: float = 0.8, top_p: float = 0.95):
        self.model_args = ModelArgs()
        self.generator = Transformer(self.model_args)
        self.temperature = temperature
        self.top_p = top_p
    
    def get_module(self):
        return self.generator 
    
    def train(self):
        return NotImplementedError

    def eval(self):
        return NotImplementedError

if __name__ == "__main__":
    model = Model()
    module = model.get_module()
    input_tensor = torch.tensor([[5, 1, 1], [1,1]], dtype=torch.int)

    module(input_tensor, 1)