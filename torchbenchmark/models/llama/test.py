
import torch
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import torch

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class Model:
    def __init__(self, temperature: float = 0.8, top_p: float = 0.95):
        self.model_args = ModelArgs()
        self.generator = Transformer(self.model_args)
        self.temperature = temperature
        self.top_p = top_p

        

    # def inference(self, prompts : str):
    #     prompts = ["The capital of Germany is the city of", "Here is my sonnet in the style of Shakespeare about an artificial intelligence:"]
    #     results = self.generator.generate(prompts, max_gen_len=256, temperature=self.temperature, top_p=self.top_p)

    #     for result in results:
    #         print(result)
    #         print("\n==================================\n")
    
    def get_module(self):
        return self.generator 
    
    def train(self):
        return NotImplementedError

    def eval(self):
        return NotImplementedError

if __name__ == "__main__":
    model = Model()
    model.get_module().generate((torch.randn(1,1,1,1)))