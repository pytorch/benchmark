from ...util.model import BenchmarkModel
from torchbenchmark.tasks import NLP
import torch


class Model(BenchmarkModel):
    task = NLP.GENERATION
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        from .model import GPTConfig, GPT
        from .model_fine import FineGPT, FineGPTConfig

        gptconf = GPTConfig()
        self.model_coarse = GPT(gptconf)

        finegptconf = FineGPTConfig()
        self.model_fine = FineGPT(finegptconf)

        # Use the default configs
        self.gpt_config = GPTConfig()
        self.generator_config = GPTGenerationConfig(500, 0.8, 200)
        self.model = SequenceGeneratorNanoGPT(GPT(self.gpt_config), self.generator_config).eval().to(self.device)
        self.prompt_size = 64
        self.example_inputs = (
            torch.randint(1, self.gpt_config.vocab_size, (self.batch_size, self.prompt_size)).to(self.device),
        )
        self.text_prompt = """
             Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
             But I also have other interests such as playing tic tac toe.
        """

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        return NotImplementedError("Training not supported for this model")

    def eval(self):

        coarse_tokens = generate_coarse(
            semantic_tokens,
            history_prompt=history_prompt,
            temp=temp,
            silent=silent,
            use_kv_caching=True
        )
        fine_tokens = generate_fine(
            coarse_tokens,
            history_prompt=history_prompt,
            temp=0.5,
        )
        return fine_tokens
        
