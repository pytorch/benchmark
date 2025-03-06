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
            name="hf_Qwen2",
            test=test,
            device=device,
            batch_size=batch_size,
            extra_args=extra_args,
        )
        
        prompt = "What is the best way to debug python script?"
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
        inputs = tokenizer(prompt, return_tensors="pt")

        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()

        self.example_inputs = {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "past_key_values": DynamicCache(), 
            "use_cache": True
        }
        self.model.to(self.device)

    def train(self):
        raise NotImplementedError("Training is not implemented.")

    def get_module(self):
        return self.model, self.example_inputs

    def eval(self):
        example_inputs = self.example_inputs
        self.model.eval()
        self.model(input_ids=example_inputs["input_ids"], attention_mask=example_inputs["attention_mask"], past_key_values=DynamicCache(), use_cache=True)
