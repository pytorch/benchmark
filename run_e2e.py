import time
import torch
import argparse
from torchbenchmark.e2e import E2EBenchmarkResult, load_e2e_model_by_name
from typing import Dict

def run(func) -> Dict[str, float]:
    torch.cuda.synchronize()
    result = {}
    # Collect time_ns() instead of time() which does not provide better precision than 1
    # second according to https://docs.python.org/3/library/time.html#time.time.
    t0 = time.time_ns()
    func()
    torch.cuda.synchronize()
    t2 = time.time_ns()
    result["latency"] = (t2 - t0) / 1_000_000_000.0
    return result

def gen_result(m, run_result):
    r = E2EBenchmarkResult()
    r.device = m.device
    r.device_num = m.device_num
    r.test = m.test
    r.examples = m.examples
    r.batch_size = m.bs
    r.latency = run_result["latency"] / 1000.0
    r.qps = r.examples / (r.latency / 1000.0)
    r.flops = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    SUPPORT_DEVICE_LIST = ["cpu", "cuda"]
    parser.add_argument("model", help="Full name of a model to run.")
    parser.add_argument("-d", "--device", choices=SUPPORT_DEVICE_LIST, default="cpu", help="Which device to use.")
    parser.add_argument("-t", "--test", choices=["eval", "train"], default="eval", help="Which test to run.")
    parser.add_argument("--bs", type=int, help="Specify batch size to the test.")
    args, extra_args = parser.parse_known_args()

    found = False
    Model = load_e2e_model_by_name(args.model)
    if not Model:
        print(f"Unable to find model matching {args.model}.")
        exit(-1)
    m = Model(device=args.device, test=args.test, bs=args.bs, extra_args=extra_args)
    test = getattr(m, args.test)
    result = gen_result(m, run(test))
