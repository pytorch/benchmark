#!/usr/bin/env python
from collections import Counter, defaultdict
from functools import partial
from typing import Any, Dict, Callable, Optional
import argparse
import gc
import logging
import os
import pandas as pd
import re
import time
import warnings

os.environ["FX_PATCH_GETITEM"] = "1"  # make BERT fx.symbolic_trace

from torchbenchmark import list_models
from torch.fx import symbolic_trace, Node, GraphModule
from torch.fx.interpreter import Interpreter
import torch

# These do not fx.symbolic_trace()
SKIP = {"attention_is_all_you_need_pytorch", "demucs", "dlrm", "maml",
        "yolov3", "tacotron2", "moco", "Super_SloMo"}


class ProfileStats(object):
    @staticmethod
    def _norm(cnt: Counter):
        """ Normalize to unit length """
        total = sum(cnt.values())
        return Counter({k: v / total for k, v in cnt.items()})

    def __init__(self, get_name: Optional[Callable]):
        super(ProfileStats, self).__init__()
        self.times: Dict[str, float] = Counter()
        self.counts: Dict[str, int] = Counter()
        self.get_name = get_name

    def record(self, node: Node, sec: float):
        """ Record timings of a single call """
        name = self.get_name(node)
        self.times[name] += sec
        self.counts[name] += 1

    def summary(self, n=5):
        most_common = self._norm(self.times).most_common(n - 1)
        return " ".join([f"{k}:{v:.0%}" for k, v in most_common] +
                        [f"other:{1.0 - sum(v for k, v in most_common):.0%}"])


class ProfileAggregate(ProfileStats):
    def __init__(self, name: str):
        super(ProfileAggregate, self).__init__(None)
        self.df = pd.DataFrame()
        self.name = name

    def update(self, other: ProfileStats, name):
        """ Merge stats from a finished benchmark run into this """
        nt = self._norm(other.times).most_common(None)
        self.times.update(nt)
        self.counts.update(self._norm(other.counts))
        self.df = self.df.append(pd.DataFrame(
            [[t for n, t in nt]],
            index=[name],
            columns=[n for n, t in nt],
        ))

    def save(self):
        df = self.df.fillna(0.0).transpose()
        df.insert(0, "AVERAGE", df.mean(axis=1))
        df.sort_values("AVERAGE", ascending=False, inplace=True)
        df.to_csv(f"{self.name}.csv")
        print(f"wrote {self.name}.csv")


PROFILES = [
    ProfileAggregate("operators"),
    ProfileAggregate("successors1"),
    ProfileAggregate("successors2"),
    ProfileAggregate("predecessors1"),
    ProfileAggregate("predecessors2"),
]


class FXProfiler(Interpreter):
    def __init__(self, module: GraphModule):
        super(FXProfiler, self).__init__(module)
        self.profile_stats = [
            ProfileStats(self.get_name),
            ProfileStats(partial(self.succ_name, depth=2)),
            ProfileStats(partial(self.succ_name, depth=3)),
            ProfileStats(partial(self.pred_name, depth=2)),
            ProfileStats(partial(self.pred_name, depth=3)),
        ]

        self.successors = defaultdict(list)
        self.predecessors = defaultdict(list)
        for node in self.module.graph.nodes:
            def visit(other_node):
                self.successors[other_node].append(node)
                self.predecessors[node].append(other_node)

            torch.fx.map_arg((node.args, node.kwargs), visit)

    def run_node(self, n: Node) -> Any:
        """ Timing wrapper around executing an FX Node """
        start = time.perf_counter()
        result = super().run_node(n)
        torch.cuda.synchronize()
        sec = time.perf_counter() - start
        for prof in self.profile_stats:
            prof.record(n, sec)
        return result

    _op_node_to_name = {
        "call_function": lambda i, t: t.__name__,
        "call_method": lambda i, t: t,
        "call_module": lambda i, t: type(i.fetch_attr(t)).__name__,
        "get_attr": lambda i, t: "get_attr",
        "output": lambda i, t: "output",
        "placeholder": lambda i, t: "placeholder",
    }

    def get_name(self, n: Node) -> Callable:
        """ Coverts a Node to a string name """
        return self._op_node_to_name[n.op](self, n.target).lower()

    def pred_name(self, node: Node, depth: int) -> Callable:
        """ A string name that includes names of predecessor nodes """
        if depth <= 1:
            return self.get_name(node)
        pred_str = ','.join(self.pred_name(x, depth - 1) for x in self.predecessors[node])
        return f"{self.get_name(node)}({pred_str})"

    def succ_name(self, node: Node, depth: int) -> Callable:
        """ A string name that includes names of successor nodes """
        s = self.successors[node]
        if depth <= 1 or len(s) == 0:
            return self.get_name(node)
        elif len(s) > 1:
            succ_str = "MANY"
        else:
            succ_str = self.succ_name(s[0], depth - 1)
        return f"{self.get_name(node)}->{succ_str}"


def profile(device, name, model, example_inputs, args):
    model = torch.fx.symbolic_trace(model)
    prof = FXProfiler(model)

    for _ in range(args.warmup):
        model(*example_inputs)

    for _ in range(args.repeat):
        torch.cuda.synchronize()
        prof.run(*example_inputs)

    for aggregate, stats in zip(PROFILES, prof.profile_stats):
        print(f"{device:4} {name:20} {aggregate.name:13} {stats.summary()}")
        aggregate.update(stats, name=name)
    return model


def short_name(name, limit=20):
    """ Truncate a model name to limit chars"""
    return name if len(name) <= limit else f"{name[:limit - 3].rstrip('_')}..."


def iter_models(args):
    for benchmark_cls in list_models():
        if (not re.search("|".join(args.filter), benchmark_cls.name, re.I) or
                re.search("|".join(args.exclude), benchmark_cls.name, re.I) or
                benchmark_cls.name in SKIP):
            continue
        try:
            benchmark = benchmark_cls(device=args.device, jit=False)
            model, example_inputs = benchmark.get_module()
            model.eval()
            gc.collect()
            yield short_name(benchmark.name), model, example_inputs
        except NotImplementedError:
            pass


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", "-k", action="append",
                        help="filter benchmarks")
    parser.add_argument("--exclude", "-x", action="append",
                        help="filter benchmarks")
    parser.add_argument("--device", "-d", help="cpu or cuda")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warmup runs to do")
    parser.add_argument("--repeat", "-n", type=int, default=10,
                        help="number of timing runs")
    parser.add_argument("--threads", "-p", type=int,
                        help="number threads")
    parser.add_argument("--cpu-fusion", action="store_true",
                        help="enable can_fuse_on_cpu")
    parser.add_argument("--no-skip", "-a", action="store_true",
                        help="run models that don't fx cleanly")
    args = parser.parse_args(args)

    # defaults
    args.device = args.device or "cpu"
    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]

    if args.no_skip:
        SKIP.clear()

    if args.cpu_fusion:
        torch._C._jit_override_can_fuse_on_cpu(True)

    if args.threads:
        torch.set_num_threads(args.threads)

    for name, model, example_inputs in iter_models(args):
        profile(args.device, name, model, example_inputs, args)

    for prof in PROFILES:
        print("SUMMARY", prof.name, prof.summary(10))
        prof.save()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
