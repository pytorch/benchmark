# Extended timm model configs from Dynamobench
import os
from typing import List

import torch
from userbenchmark.dynamo import DYNAMOBENCH_PATH

TIMM_MODELS = dict()
# Only load the extended models in OSS
if hasattr(torch.version, "git_version"):
    filename = os.path.join(DYNAMOBENCH_PATH, "timm_models_list.txt")
    with open(filename) as fh:
        lines = fh.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            model_name, batch_size = line.split(" ")
            TIMM_MODELS[model_name] = int(batch_size)


def is_extended_timm_models(model_name: str) -> bool:
    return model_name in TIMM_MODELS


def list_extended_timm_models() -> List[str]:
    return list(TIMM_MODELS.keys())


# TODO - Figure out the reason of cold start memory spike
BATCH_SIZE_DIVISORS = {
    "beit_base_patch16_224": 2,
    "cait_m36_384": 8,
    "convit_base": 2,
    "convmixer_768_32": 2,
    "convnext_base": 2,
    "cspdarknet53": 2,
    "deit_base_distilled_patch16_224": 2,
    "dpn107": 2,
    "gluon_xception65": 2,
    "mobilevit_s": 2,
    "pit_b_224": 2,
    "pnasnet5large": 2,
    "poolformer_m36": 2,
    "res2net101_26w_4s": 2,
    "resnest101e": 2,
    "sebotnet33ts_256": 2,
    "swin_base_patch4_window7_224": 2,
    "swsl_resnext101_32x16d": 2,
    "twins_pcpvt_base": 2,
    "vit_base_patch16_224": 2,
    "volo_d1_224": 2,
    "jx_nest_base": 4,
    "xcit_large_24_p8_224": 4,
}

REQUIRE_HIGHER_TOLERANCE = {
    "fbnetv3_b",
    "gmixer_24_224",
    "hrnet_w18",
    "inception_v3",
    "sebotnet33ts_256",
    "selecsls42b",
}
REQUIRE_HIGHER_TOLERANCE_FOR_FREEZING = {
    "adv_inception_v3",
    "botnet26t_256",
    "gluon_inception_v3",
    "selecsls42b",
    "swsl_resnext101_32x16d",
}

SCALED_COMPUTE_LOSS = {
    "ese_vovnet19b_dw",
    "fbnetc_100",
    "mnasnet_100",
    "mobilevit_s",
    "sebotnet33ts_256",
}

FORCE_AMP_FOR_FP16_BF16_MODELS = {
    "convit_base",
    "xcit_large_24_p8_224",
}

SKIP_ACCURACY_CHECK_AS_EAGER_NON_DETERMINISTIC_MODELS = {
    "xcit_large_24_p8_224",
}
