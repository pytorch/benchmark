from typing import Dict

# NVIDIA A100 GPU Spec:
# https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
NV_A100 = {
    "fp32": 19.5,
    "tf32": 156,
    "bf16": 312,
    "fp16": 312,
}

# NVIDIA H100 GPU Datasheet:
# https://nvdam.widen.net/content/vuzumiozpb/original/h100-datasheet-2287922.pdf
NV_H100 = {
    "fp32": 51,
    "tf32": 756,
    "bf16": 1513,
    "fp16": 1513,
}


HW_ROOFLINE_SPECS: Dict[str, Dict[str, float]] = {
    "NVIDIA A100-SXM4-40GB": NV_A100,
    "NVIDIA A100-PG509-200": NV_A100,
    "NVIDIA H100": NV_H100,
}
