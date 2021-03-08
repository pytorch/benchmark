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

def get_dump_filename(name, device, args):
    if args.no_profiling:
        return f"{name}.{device}.last_executed_graph.noprofile.log"
    if args.inlined_graph:
        return f"{name}.{device}.inlined_graph.log"
    return f"{name}.{device}.last_executed_graph.log"

def iter_models(args):
    device = "cpu"
    for benchmark_cls in list_models():
        bench_name = benchmark_cls.name
        if args.benchmark and args.benchmark != bench_name:
            continue
        if bench_name in NO_GET_MODULE:
            print(f"{bench_name} has no get_module, skipped")
            continue
        if bench_name in NO_JIT:
            print(f"{bench_name} has no scripted module, skipped")
            continue
        try:
            # disable profiling mode so that the collected graph does not contain
            # profiling node
            if args.no_profiling:
                torch._C._jit_set_profiling_mode(False)

            benchmark = benchmark_cls(device=device, jit=True)
            model, example_inputs = benchmark.get_module()

            # extract ScriptedModule object for BERT model
            if bench_name == "BERT_pytorch":
                model = model.bert

            fname = get_dump_filename(bench_name, device, args)
            print(f"Dump Graph IR for {bench_name} to {fname}")
            
            # default mode need to warm up ProfileExecutor
            if not (args.no_profiling or args.inlined_graph):
                model.graph_for(*example_inputs)

            with open(fname, 'w') as dump_file:
                if args.inlined_graph:
                    print(model.inlined_graph, file=dump_file)
                else:
                    print(model.graph_for(*example_inputs), file=dump_file)
        except NotImplementedError:
            print(f"Cannot collect graph IR dump for {bench_name}")
            pass

def main(args=None):
    parser = argparse.ArgumentParser(description="dump last_executed graph for all benchmarks with JIT implementation")
    parser.add_argument("--benchmark", "-b", 
                        help="dump graph for <BENCHMARK>")
    parser.add_argument("--no_profiling", action="store_true",
                        help="dump last_executed graphs w/o profiling executor")
    parser.add_argument("--inlined_graph", action="store_true",
                        help="dump graphs dumped by module.inlined_graph")
    args = parser.parse_args(args)

    iter_models(args)

if __name__ == '__main__':
    main()
