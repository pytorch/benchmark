import torch
from torchbenchmark.tasks import NLP
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceModel
from transformers import AutoTokenizer, DynamicCache, AutoModelForCausalLM


class Model(HuggingFaceModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_EVAL_BSIZE = 1
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    def __init__(self, test="inference", device="cuda", batch_size=None, extra_args=[]):
        # self.device = "cuda"
        super().__init__(
            name="hf_simplescaling",
            test=test,
            device=device,
            batch_size=batch_size,
            extra_args=extra_args,
        )
        
        tokenizer = AutoTokenizer.from_pretrained("simplescaling/s1.1-32B")

        prompt = "How many r in raspberry"
        messages = [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
        self.example_inputs = {**model_inputs, "max_new_tokens":512}

        class WrapperModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                return self.model.generate(*args, **kwargs)

        self.model = WrapperModel(self.model)

    def train(self):
        raise NotImplementedError("Training is not implemented.")

    def get_module(self):
        return self.model, self.example_inputs

    def eval(self):
        example_inputs = self.example_inputs
        self.model.eval()
        self.model(**example_inputs)
