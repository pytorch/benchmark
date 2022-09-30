import argparse
from typing import List, Dict
from torchbenchmark import _list_model_paths, ModelTask
from ..utils import dump_output, get_output_dir,

TIMEOUT = 300  # Seconds
BEFORE_CUDA_MEM = 0
AFTER_CUDA_MEM = 0
BM_NAME = "cuda-leak-test"

def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BERT_pytorch", help="Name of the model to test CUDA memory leak.")

def _create_example_model_instance(task: ModelTask, device: str):
    skip = False
    try:
        task.make_model_instance(test="eval", device=device, jit=False)
    except NotImplementedError:
        try:
            task.make_model_instance(test="train", device=device, jit=False)
        except NotImplementedError:
            skip = True
    finally:
        if skip:
            raise NotImplementedError(f"Model is not implemented on the device {device}")

def assert_equal(before, after):
    BEFORE_CUDA_MEM = before
    AFTER_CUDA_MEM = after

def test_cuda_memory(path: str) -> Dict[str, int]:
    task = ModelTask(path, timeout=TIMEOUT)
    out = {}
    device = "cuda"
    with task.watch_cuda_memory(skip=False, assert_equal=assert_equal):
        _create_example_model_instance(task, device)
        out["model_name"] = task.get_model_attribute("name")
        task.check_example()
        task.del_model_instance()
    out["before_model_mem"] = BEFORE_CUDA_MEM
    out["after_model_mem"] = AFTER_CUDA_MEM
    return out

def gen_output_metrics(cuda_memory_profile):
    pass

def run(args: List[str]):
    args = parse_args(args)
    paths = _list_model_paths()
    output_dir = get_output_dir(BM_NAME)
    test_paths = filter(lambda x: args.model, paths)
    for test_path in test_paths:
        cuda_memory_profile = test_cuda_memory(test_path)
        gen_output_metrics(cuda_memory_profile)
