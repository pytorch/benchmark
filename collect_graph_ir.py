#!/usr/bin/env python
import argparse
import gc
import logging
import os
import re
import warnings

from torchbenchmark import list_models
import torch

NO_JIT = {"demucs", "dlrm", "maml", "yolov3", "moco", "pytorch_CycleGAN_and_pix2pix", "tacotron2"}
NO_GET_MODULE = {"Background_Matting"}

def get_dump_filename(name, device):
    return f"{name}.{device}.last_executed_graph_dump.log"

def iter_models(args):
    device = "cpu"
    for benchmark_cls in list_models():
        #if (not re.search("|".join(args.filter), benchmark_cls.name, re.I) or
        #        re.search("|".join(args.exclude), benchmark_cls.name, re.I) or
        if benchmark_cls.name in NO_GET_MODULE:
            print(f"{benchmark_cls.name} has no get_module, skipped")
            continue
        if benchmark_cls.name in NO_JIT:
            print(f"{benchmark_cls.name} has no scripted module, skipped")
            continue
        try:
            # disable profiling mode so that the collected graph does not contain
            # profiling node
            torch._C._jit_set_profiling_mode(False)

            benchmark = benchmark_cls(device=device, jit=True)
            model, example_inputs = benchmark.get_module()

            if benchmark_cls.name == "BERT_pytorch":
                # extract ScriptedModule object for BERT model
                model = model.bert

            fname = get_dump_filename(benchmark.name, device)
            print(f"Dump Graph IR for {benchmark.name} to {fname}")
            with open(fname, 'w') as dump_file:
                print(model.graph_for(*example_inputs), file=dump_file)
        except NotImplementedError:
            print(f"Cannot collect graph IR dump for {benchmark.name}")
            pass

def main(args=None):
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--filter", "-k", action="append",
    #                    help="filter benchmarks")
    #args = parser.parse_args(args)

    iter_models(args)

if __name__ == '__main__':
    main()
