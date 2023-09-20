from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch
from .model import GPT, GPTConfig


class Model(BenchmarkModel):
    task = NLP.GENERATION
    DEFAULT_TRAIN_BSIZE = 1

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        # Use the default configs
        # self.gpt_config = GPTConfig()
        # self.generator_config = GPTGenerationConfig(32, 0.8, 200)
        # self.model = SequenceGeneratorNanoGPT(GPT(self.gpt_config), self.generator_config).eval().to(self.device)
        # self.prompt_size = 64
        # self.example_inputs = (
        #     torch.randint(1, self.gpt_config.vocab_size, (self.batch_size, self.prompt_size)).to(self.device),
        # )

        # from scratch
        batch_size = 12

        n_layer = 12
        n_head = 12
        n_embd = 768
        block_size = 1024
        bias = False
        vocab_size=50304
        dropout = 0.0
        prompt_size = 64
        learning_rate = 6e-4
        weight_decay = 1e-1
        beta1 = 0.9
        beta2 = 0.95
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
        gptconf = GPTConfig(**model_args)

        self.model = GPT(gptconf).to(device)
        self.optimizer = self.model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
        self.example_inputs = (
            torch.randint(1, gptconf.vocab_size, (batch_size, prompt_size)).to(self.device),
        )
        self.target_inputs = torch.randint(1, gptconf.vocab_size, (batch_size, prompt_size)).to(self.device)

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        _, loss = self.model(*self.example_inputs, self.target_inputs)
        loss.backward()
        self.optimizer.step()

    def eval(self):
        return NotImplementedError("Inference not supported for this model")
