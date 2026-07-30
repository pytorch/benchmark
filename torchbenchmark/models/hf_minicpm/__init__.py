import torch
from torchbenchmark.tasks import NLP
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceModel
from transformers import AutoTokenizer, DynamicCache, AutoModelForCausalLM
import librosa
from contextlib import contextmanager
from pathlib import Path  
import torch.utils._pytree as pytree

def copy_tensors(inputs):
    return pytree.tree_map_only(torch.Tensor, torch.clone, inputs)

def add_sampling_hook(module, samples, hooks):
    def _(module, args, kwargs):
        print("INSIDE HOOK")
        samples.append(copy_tensors((args, kwargs)))

    hook = module.register_forward_pre_hook(_, prepend=True, with_kwargs=True)
    hooks.append(hook)
    return hook


class Model(HuggingFaceModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_EVAL_BSIZE = 1
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    def __init__(self, test="inference", device="cuda", batch_size=None, extra_args=[]):
        # self.device = "cuda"
        super().__init__(
            name="hf_minicpm",
            test=test,
            device=device,
            batch_size=batch_size,
            extra_args=extra_args,
        )

        
        prompt = "What is the best way to debug python script?"
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)
        inputs = tokenizer(prompt, return_tensors="pt")

        self.model.init_tts()
        self.model.tts.float()

        class WrapperModule(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                return self.model.generate(*args, **kwargs)
        
        self.model = WrapperModule(self.model.tts)
        
        self.example_inputs = ((), {
            "input_ids": torch.zeros((1, 303, 4), device=self.device), 
            "past_key_values": [
                (
                    torch.randn((1, self.model.model.config.num_attention_heads, 302, 64), device=self.device),
                    torch.randn((1, self.model.model.config.num_attention_heads, 302, 64), device=self.device),
                ) for _ in range(self.model.model.config.num_hidden_layers)
            ],
            "temperature": torch.tensor([0.1000, 0.3000, 0.1000, 0.3000], device=self.device),
            "eos_token": torch.tensor([625], device=self.device),
            "streaming_tts_text_mask": torch.ones([303], dtype=torch.int8, device=self.device),
            "max_new_token": 25,
        })

        self.model.to(self.device)

    def train(self):
        raise NotImplementedError("Training is not implemented.")

    def get_module(self):
        return self.model, self.example_inputs

    def eval(self):
        example_inputs_args, example_inputs_kwargs = self.example_inputs
        example_inputs_kwargs["past_key_values"] = DynamicCache()   
        self.model.eval()
        self.model(*example_inputs_args, **example_inputs_kwargs)
