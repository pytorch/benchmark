import json
import torch
import platform
import os
import time
import datetime
import hashlib

def get_arch_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    else:
        # This returns x86_64 or arm64 (for aarch64)
        return platform.machine()


def write_json_result(output_json_path, headers, row):
    """
    Write the result into JSON format, so that it can be uploaded to the benchmark database
    to be displayed on OSS dashboard. The JSON format is defined at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    mapping_headers = {headers[i]: v for i, v in enumerate(row)}
    today = datetime.date.today()
    sha_hash = hashlib.sha256(str(today).encode("utf-8")).hexdigest()
    first_second = datetime.datetime.combine(today, datetime.time.min)
    workflow_id = int(first_second.timestamp())
    job_id = workflow_id + 1
    record = {
        "timestamp": int(time.time()),
        "schema_version": "v3",
        "name": "devvm local benchmark",
        "repo": "pytorch/ao",
        "head_branch": "main",
        "head_sha": sha_hash,
        "workflow_id": workflow_id,
        "run_attempt": 1,
        "job_id": job_id,
        "benchmark": {
            "name": "TorchAO benchmark",
            "mode": "inference",
            "dtype": mapping_headers["dtype"],
            "extra_info": {
                "device": mapping_headers["device"],
                "arch": mapping_headers["arch"],
                "min_sqnr": None,
                "compile": mapping_headers["compile"],
            },
        },
        "model": {
            "name": mapping_headers["name"],
            "type": "model",
            # TODO: make this configurable
            "origins": ["torchbench"],
        },
        "metric": {
            "name": mapping_headers["metric"],
            "benchmark_values": [mapping_headers["actual"]],
            "target_value": mapping_headers["target"],
        },
    }

    with open(f"{os.path.splitext(output_json_path)[0]}.json", "a") as f:
        print(json.dumps(record), file=f)

def benchmark_and_write_json_result(model, args, kwargs, quantization, device, compile=True):
    print(quantization + " run")
    from torchao.utils import benchmark_model, profiler_runner
    if compile:
        model = torch.compile(model, mode="max-autotune")
    benchmark_model(model, 20, args, kwargs)
    elapsed_time = benchmark_model(model, 100, args, kwargs)
    print("elapsed_time: ", elapsed_time, " milliseconds")

    if hasattr(model, "_orig_mod"):
        name = model._orig_mod.__class__.__name__
    else:
        # eager
        name = model.__class__.__name__

    headers = ["name", "dtype", "compile", "device", "arch", "metric", "actual", "target"]
    arch = get_arch_name()
    dtype = quantization
    performance_result = [name, dtype, compile, device, arch, "time_ms(avg)", elapsed_time, None]
    _OUTPUT_JSON_PATH = "benchmark_results"
    write_json_result(_OUTPUT_JSON_PATH, headers, performance_result)
