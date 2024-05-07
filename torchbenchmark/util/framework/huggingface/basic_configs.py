import os
import re
from typing import List

import torch
import transformers

HUGGINGFACE_MODELS = {
    # 'name': (train_max_length, eval_max_length, config, model)
    "hf_GPT2": (
        512,
        1024,
        'AutoConfig.from_pretrained("gpt2")',
        "AutoModelForCausalLM",
    ),
    "hf_GPT2_large": (
        512,
        1024,
        'AutoConfig.from_pretrained("gpt2-large")',
        "AutoModelForCausalLM",
    ),
    "hf_T5": (
        1024,
        2048,
        'AutoConfig.from_pretrained("t5-small")',
        "AutoModelForSeq2SeqLM",
    ),
    "hf_T5_base": (
        1024,
        2048,
        'AutoConfig.from_pretrained("t5-base")',
        "AutoModelForSeq2SeqLM",
    ),
    "hf_T5_large": (
        512,
        512,
        'AutoConfig.from_pretrained("t5-large")',
        "AutoModelForSeq2SeqLM",
    ),
    "hf_Bart": (
        512,
        512,
        'AutoConfig.from_pretrained("facebook/bart-base")',
        "AutoModelForSeq2SeqLM",
    ),
    "hf_Reformer": (
        4096,
        4096,
        "ReformerConfig(num_buckets=128)",
        "AutoModelForMaskedLM",
    ),
    "hf_BigBird": (
        1024,
        4096,
        'BigBirdConfig(attention_type="block_sparse",)',
        "AutoModelForMaskedLM",
    ),
    "hf_Albert": (
        512,
        512,
        'AutoConfig.from_pretrained("albert-base-v2")',
        "AutoModelForMaskedLM",
    ),
    "hf_DistilBert": (
        512,
        512,
        'AutoConfig.from_pretrained("distilbert-base-uncased")',
        "AutoModelForMaskedLM",
    ),
    "hf_Longformer": (
        1024,
        4096,
        'AutoConfig.from_pretrained("allenai/longformer-base-4096")',
        "AutoModelForMaskedLM",
    ),
    "hf_Bert": (512, 512, "BertConfig()", "AutoModelForMaskedLM"),
    # see https://huggingface.co/bert-large-cased
    "hf_Bert_large": (
        512,
        512,
        "BertConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16)",
        "AutoModelForMaskedLM",
    ),
    "hf_Whisper": (1024, 1024, "WhisperConfig()", "AutoModelForAudioClassification"),
    "hf_distil_whisper": (
        1024,
        1024,
        'AutoConfig.from_pretrained("distil-whisper/distil-medium.en")',
        "AutoModelForAudioClassification",
    ),
    "hf_mixtral": (
        512,
        512,
        'AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-v0.1")',
        "AutoModelForCausalLM",
    ),
    # default num_hidden_layers=32 but that OOMs, feel free to change this config to something more real
    "llama_v2_7b_16h": (
        128,
        512,
        "LlamaConfig(num_hidden_layers=16)",
        "AutoModelForCausalLM",
    ),
    "hf_MPT_7b_instruct": (
        512,
        512,
        'AutoConfig.from_pretrained("mosaicml/mpt-7b-instruct", trust_remote_code=True)',
        "AutoModelForCausalLM",
    ),
    "llava": (
        512,
        512,
        'AutoConfig.from_pretrained("liuhaotian/llava-v1.5-13b")',
        "LlavaForConditionalGeneration",
    ),
    "llama_v2_7b": (
        512,
        512,
        'AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")',
        "AutoModelForCausalLM",
    ),
    "llama_v2_13b": (
        512,
        512,
        'AutoConfig.from_pretrained("meta-llama/Llama-2-13b-hf")',
        "AutoModelForCausalLM",
    ),
    "llama_v2_70b": (
        512,
        512,
        'AutoConfig.from_pretrained("meta-llama/Llama-2-70b-hf")',
        "AutoModelForMaskedLM",
    ),
    "codellama": (
        512,
        512,
        'AutoConfig.from_pretrained("codellama/CodeLlama-7b-hf")',
        "AutoModelForCausalLM",
    ),
    "phi_1_5": (
        512,
        512,
        'AutoConfig.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)',
        "AutoModelForCausalLM",
    ),
    "phi_2": (
        512,
        512,
        'AutoConfig.from_pretrained("microsoft/phi-2", trust_remote_code=True)',
        "AutoModelForCausalLM",
    ),
    "moondream": (
        512,
        512,
        'PhiConfig.from_pretrained("vikhyatk/moondream1")',
        "PhiForCausalLM",
    ),
    # as per this page https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 trust_remote_code=True is not required
    "mistral_7b_instruct": (
        128,
        128,
        'AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")',
        "AutoModelForCausalLM",
    ),
    "hf_Yi": (
        512,
        512,
        'AutoConfig.from_pretrained("01-ai/Yi-6B", trust_remote_code=True)',
        "AutoModelForCausalLM",
    ),
    "orca_2": (
        512,
        512,
        'AutoConfig.from_pretrained("microsoft/Orca-2-13b")',
        "AutoModelForCausalLM",
    ),
}

CPU_INPUT_SLICE = {
    "hf_BigBird": 5,
    "hf_Longformer": 8,
    "hf_T5": 4,
    "hf_GPT2": 4,
    "hf_Reformer": 2,
}

HUGGINGFACE_MODELS_REQUIRING_TRUST_REMOTE_CODE = [
    "hf_Falcon_7b",
    "hf_MPT_7b_instruct",
    "phi_1_5",
    "phi_2",
    "hf_Yi",
    "hf_mixtral",
]

HUGGINGFACE_MODELS_SGD_OPTIMIZER = [
    "llama_v2_7b_16h",
]


def is_basic_huggingface_models(model_name: str) -> bool:
    return model_name in HUGGINGFACE_MODELS


def list_basic_huggingface_models() -> List[str]:
    return HUGGINGFACE_MODELS.keys()


def generate_inputs_for_model(
    model_cls,
    model,
    model_name,
    bs,
    device,
    is_training=False,
):
    if is_training:
        max_length = HUGGINGFACE_MODELS[model_name][0]
    else:
        max_length = HUGGINGFACE_MODELS[model_name][1]
    # populate these on-demand to avoid wasting memory when not used
    if is_training:
        input_ids = torch.randint(0, model.config.vocab_size, (bs, max_length)).to(
            device
        )
        decoder_ids = torch.randint(0, model.config.vocab_size, (bs, max_length)).to(
            device
        )
        example_inputs = {"input_ids": input_ids, "labels": decoder_ids}
    else:
        # Cut the length of sentence when running on CPU, to reduce test time
        if device == "cpu" and model_name in CPU_INPUT_SLICE:
            max_length = int(max_length / CPU_INPUT_SLICE[model_name])
        eval_context = torch.randint(0, model.config.vocab_size, (bs, max_length)).to(
            device
        )
        example_inputs = {
            "input_ids": eval_context,
        }
        if model_cls.__name__ in ["AutoModelForSeq2SeqLM"]:
            example_inputs["decoder_input_ids"] = eval_context
    return example_inputs


def generate_input_iter_for_model(
    model_cls,
    model,
    model_name,
    bs,
    device,
    is_training=False,
):
    import math
    import random

    nbuckets = 8
    if is_training:
        max_length = HUGGINGFACE_MODELS[model_name][0]
    else:
        max_length = HUGGINGFACE_MODELS[model_name][1]
    n = int(math.log2(max_length))
    buckets = [2**n for n in range(n - nbuckets, n)]
    if model_cls.__name__ == "AutoModelForSeq2SeqLM":
        raise NotImplementedError("AutoModelForSeq2SeqLM is not yet supported")
    while True:
        # randomize bucket_len
        bucket_len = random.choice(buckets)
        dict_input = {
            "input_ids": torch.randint(0, model.config.vocab_size, (bs, bucket_len)).to(
                device
            ),
            "labels": torch.randint(0, model.config.vocab_size, (bs, bucket_len)).to(
                device
            ),
        }
        yield dict_input


def download_model(model_name):
    def _extract_config_cls_name(config_cls_ctor: str) -> str:
        """Extract the class name from the given string of config object creation.
        For example,
        if the constructor runs like `AutoConfig.from_pretrained("gpt2")`, return "AutoConfig".
        if the constructor runs like `LlamaConfig(num_hidden_layers=16)`, return "LlamaConfig".
        """
        pattern = r"([A-Za-z0-9_]*)[\(\.].*"
        m = re.match(pattern, config_cls_ctor)
        return m.groups()[0]

    config_cls_name = _extract_config_cls_name(HUGGINGFACE_MODELS[model_name][2])
    exec(f"from transformers import {config_cls_name}")
    config = eval(HUGGINGFACE_MODELS[model_name][2])
    model_cls = getattr(transformers, HUGGINGFACE_MODELS[model_name][3])
    kwargs = {}
    if model_name in HUGGINGFACE_MODELS_REQUIRING_TRUST_REMOTE_CODE:
        kwargs["trust_remote_code"] = True
    if hasattr(model_cls, "from_config"):
        model = model_cls.from_config(config, **kwargs)
    else:
        model = model_cls(config, **kwargs)
    return model_cls, model


def generate_optimizer_for_model(model, model_name):
    from torch import optim

    if model_name in HUGGINGFACE_MODELS_SGD_OPTIMIZER:
        return optim.SGD(model.parameters(), lr=0.001)
    return optim.Adam(
        model.parameters(),
        lr=0.001,
        # TODO resolve https://github.com/pytorch/torchdynamo/issues/1083
        capturable=bool(int(os.getenv("ADAM_CAPTURABLE", 0))),
    )
