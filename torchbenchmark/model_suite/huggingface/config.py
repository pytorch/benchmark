import logging
import torch
import importlib

imports = [
    "AlbertForPreTraining",
    "AutoConfig",
    "AutoModelForCausalLM",
    "AutoModelForMaskedLM",
    "AutoModelForSeq2SeqLM",
    "BigBirdConfig",
    "BlenderbotForConditionalGeneration",
    "BlenderbotModel",
    "BlenderbotSmallForConditionalGeneration",
    "BlenderbotSmallModel",
    "CLIPModel",
    "CLIPVisionModel",
    "ElectraForPreTraining",
    "GPT2ForSequenceClassification",
    "GPTJForSequenceClassification",
    "GPTNeoForSequenceClassification",
    "HubertForSequenceClassification",
    "LxmertForPreTraining",
    "LxmertForQuestionAnswering",
    "MarianForCausalLM",
    "MarianModel",
    "MarianMTModel",
    "PegasusForConditionalGeneration",
    "PegasusModel",
    "ReformerConfig",
    "ViTForImageClassification",
    "ViTForMaskedImageModeling",
    "ViTModel",
]


mod = importlib.import_module("transformers")
for cls in imports:
    if not hasattr(mod, cls):
        raise ModuleNotFoundError


log = logging.getLogger(__name__)

# TODO - Fails even after fake tensors
BATCH_SIZE_DIVISORS = {
    "AlbertForMaskedLM": 2,
    "AlbertForQuestionAnswering": 2,
    "AllenaiLongformerBase": 2,
    "BartForCausalLM": 2,
    "BartForConditionalGeneration": 2,
    "BertForMaskedLM": 2,
    "BertForQuestionAnswering": 2,
    "BlenderbotForCausalLM": 8,
    # "BlenderbotForConditionalGeneration" : 16,
    "BlenderbotSmallForCausalLM": 4,
    "BlenderbotSmallForConditionalGeneration": 2,
    "CamemBert": 2,
    "DebertaForMaskedLM": 4,
    "DebertaForQuestionAnswering": 2,
    "DebertaV2ForMaskedLM": 4,
    "DebertaV2ForQuestionAnswering": 8,
    "DistilBertForMaskedLM": 2,
    "DistilBertForQuestionAnswering": 2,
    "DistillGPT2": 2,
    "ElectraForCausalLM": 2,
    "ElectraForQuestionAnswering": 2,
    "GPT2ForSequenceClassification": 2,
    # "GPTJForCausalLM" : 2,
    # "GPTJForQuestionAnswering" : 2,
    # "GPTNeoForCausalLM" : 32,
    # "GPTNeoForSequenceClassification" : 2,
    "GoogleFnet": 2,
    "LayoutLMForMaskedLM": 2,
    "LayoutLMForSequenceClassification": 2,
    "M2M100ForConditionalGeneration": 4,
    "MBartForCausalLM": 2,
    "MBartForConditionalGeneration": 2,
    "MT5ForConditionalGeneration": 2,
    "MegatronBertForCausalLM": 4,
    "MegatronBertForQuestionAnswering": 2,
    "MobileBertForMaskedLM": 2,
    "MobileBertForQuestionAnswering": 2,
    "OPTForCausalLM": 2,
    "PLBartForCausalLM": 2,
    "PLBartForConditionalGeneration": 2,
    "PegasusForCausalLM": 4,
    "PegasusForConditionalGeneration": 2,
    "RobertaForCausalLM": 2,
    "RobertaForQuestionAnswering": 2,
    "Speech2Text2ForCausalLM": 4,
    "T5ForConditionalGeneration": 2,
    "T5Small": 2,
    "TrOCRForCausalLM": 2,
    "XGLMForCausalLM": 4,
    "XLNetLMHeadModel": 2,
    "YituTechConvBert": 2,
}

EXTRA_MODELS = {
    "AllenaiLongformerBase": (
        AutoConfig.from_pretrained("allenai/longformer-base-4096"),
        AutoModelForMaskedLM,
    ),
    "Reformer": (
        ReformerConfig(),
        AutoModelForMaskedLM,
    ),
    "T5Small": (
        AutoConfig.from_pretrained("t5-small"),
        AutoModelForSeq2SeqLM,
    ),
    # "BigBird": (
    #     BigBirdConfig(attention_type="block_sparse"),
    #     AutoModelForMaskedLM,
    # ),
    "DistillGPT2": (
        AutoConfig.from_pretrained("distilgpt2"),
        AutoModelForCausalLM,
    ),
    "GoogleFnet": (
        AutoConfig.from_pretrained("google/fnet-base"),
        AutoModelForMaskedLM,
    ),
    "YituTechConvBert": (
        AutoConfig.from_pretrained("YituTech/conv-bert-base"),
        AutoModelForMaskedLM,
    ),
    "CamemBert": (
        AutoConfig.from_pretrained("camembert-base"),
        AutoModelForMaskedLM,
    ),
}

SKIP_ACCURACY_CHECK_MODELS = {
    # Models too large to have eager, dynamo and fp64_numbers simultaneosuly
    # even for 40 GB machine.
    "DebertaV2ForMaskedLM",
    "BlenderbotForCausalLM",
}

SKIP_DUE_TO_CONTROL_FLOW = {"AllenaiLongformerBase"}


REQUIRE_HIGHER_TOLERANCE_TRAINING = {
    "MT5ForConditionalGeneration",
    # AlbertForQuestionAnswering fails in CI GCP A100 but error does not seem
    # harmful.
    "AlbertForQuestionAnswering",
}
REQUIRE_HIGHER_TOLERANCE_INFERENCE = {
    "GPT2ForSequenceClassification",
    "RobertaForQuestionAnswering",
}


SKIP_FOR_CPU = {
    "OPTForCausalLM",  # OOMs
}

ONLY_EVAL_MODE = {
    "M2M100ForConditionalGeneration",  # Fails with dynamo for train mode
}

FP32_ONLY_MODELS = {
    "GoogleFnet",
}

def get_sequence_length(model_cls, model_name):
    if model_name.startswith(("Blenderbot",)):
        seq_length = 128
    elif model_name.startswith(("GPT2", "Bart", "T5", "PLBart", "MBart")):
        seq_length = 1024
    elif model_name in ("AllenaiLongformerBase", "BigBird"):
        seq_length = 1024
    elif model_name.startswith("OPT"):
        seq_length = 2048
    elif "Reformer" in model_name:
        seq_length = 4096
    elif model_name.startswith(
        (
            "Albert",
            "Deberta",
            "Layout",
            "Electra",
            "XLNet",
            "MegatronBert",
            "Bert",
            "Roberta",
        )
    ) or model_name in ("DistillGPT2", "GoogleFnet", "YituTechConvBert", "CamemBert"):
        seq_length = 512
    elif model_name in ("TrOCRForCausalLM"):
        seq_length = 256
    elif model_name.startswith("MobileBert"):
        seq_length = 128
    elif model_name.startswith("Wav2Vec2"):
        # If too short, will fail with something like
        # ValueError: `mask_length` has to be smaller than `sequence_length`,
        # but got `mask_length`: 10 and `sequence_length`: 9`
        seq_length = 10000  # NB: a more realistic size is 155136
    else:
        log.info(
            f"Sequence Length not defined for {model_name}. Choosing 128 arbitrarily"
        )
        seq_length = 128
    return seq_length


def rand_int_tensor(device, low, high, shape):
    return torch.randint(
        low,
        high,
        shape,
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


def generate_inputs_for_model(
    model_cls, model, model_name, bs, device, include_loss_args=False
):
    # TODO - Check if following values are representative
    num_choices = 3
    num_visual_features = 42
    seq_length = get_sequence_length(model_cls, model_name)
    vocab_size = model.config.vocab_size

    if model_name.startswith("Wav2Vec2"):
        # TODO: If we add more input_values style models, try to work this
        # into the overall control flow
        target_length = 100
        return {
            "input_values": torch.randn((bs, seq_length), device=device),
            # Added because that's what the example training script has
            "attention_mask": rand_int_tensor(device, 0, 2, (bs, seq_length)),
            "labels": rand_int_tensor(device, 0, vocab_size, (bs, target_length)),
        }

    if model_name.endswith("MultipleChoice"):
        input = rand_int_tensor(device, 0, vocab_size, (bs, num_choices, seq_length))
    elif model_name.startswith("Roberta"):
        input = rand_int_tensor(device, 0, 1, (bs, seq_length))
    else:
        input = rand_int_tensor(device, 0, vocab_size, (bs, seq_length))

    if "Bart" in model_name:
        input[:, -1] = model.config.eos_token_id

    input_dict = {"input_ids": input}

    if (
        model_name.startswith("T5")
        or model_name.startswith("M2M100")
        or model_name.startswith("MT5")
        or model_cls
        in [
            BlenderbotModel,
            BlenderbotSmallModel,
            BlenderbotForConditionalGeneration,
            BlenderbotSmallForConditionalGeneration,
            PegasusModel,
            PegasusForConditionalGeneration,
            MarianModel,
            MarianMTModel,
        ]
    ):
        input_dict["decoder_input_ids"] = input

    if model_name.startswith("Lxmert"):
        visual_feat_dim, visual_pos_dim = (
            model.config.visual_feat_dim,
            model.config.visual_pos_dim,
        )
        input_dict["visual_feats"] = torch.randn(
            bs, num_visual_features, visual_feat_dim
        )
        input_dict["visual_pos"] = torch.randn(bs, num_visual_features, visual_pos_dim)

    if include_loss_args:
        if model_name.endswith("PreTraining"):
            if model_cls in [ElectraForPreTraining, LxmertForPreTraining]:
                input_dict["labels"] = rand_int_tensor(device, 0, 1, (bs, seq_length))
            else:
                label_name = (
                    "sentence_order_label"
                    if model_cls in [AlbertForPreTraining]
                    else "next_sentence_label"
                )
                input_dict["labels"] = (
                    rand_int_tensor(device, 0, vocab_size, (bs, seq_length)),
                )
                input_dict[label_name] = rand_int_tensor(device, 0, 1, (bs,))
        elif model_name.endswith("QuestionAnswering"):
            input_dict["start_positions"] = rand_int_tensor(
                device, 0, seq_length, (bs,)
            )
            input_dict["end_positions"] = rand_int_tensor(device, 0, seq_length, (bs,))
        elif (
            model_name.endswith("MaskedLM")
            or model_name.endswith("HeadModel")
            or model_name.endswith("CausalLM")
            or model_name.endswith("DoubleHeadsModel")
        ):
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size, (bs, seq_length)
            )
        elif model_name.endswith("TokenClassification"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, model.config.num_labels - 1, (bs, seq_length)
            )
        elif model_name.endswith("MultipleChoice"):
            input_dict["labels"] = rand_int_tensor(device, 0, num_choices, (bs,))
        elif model_name.endswith("SequenceClassification"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, model.config.num_labels - 1, (bs,)
            )
        elif model_name.endswith("NextSentencePrediction"):
            input_dict["labels"] = rand_int_tensor(device, 0, 1, (bs,))
        elif model_name.endswith("ForConditionalGeneration"):
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size - 1, (bs, seq_length)
            )
        elif model_name in EXTRA_MODELS:
            input_dict["labels"] = rand_int_tensor(
                device, 0, vocab_size, (bs, seq_length)
            )
        else:
            raise NotImplementedError(
                f"Class {model_name} unsupported for training test "
            )

    return input_dict
