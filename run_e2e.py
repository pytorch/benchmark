import time
import torch
import argparse
import json
from dataclasses import asdict
from torchbenchmark.e2e import E2EBenchmarkResult, load_e2e_model_by_name
from typing import Dict

SUPPORT_DEVICE_LIST = ["cpu", "cuda"]

def run(func) -> Dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    result = {}
    # Collect time_ns() instead of time() which does not provide better precision than 1
    # second according to https://docs.python.org/3/library/time.html#time.time.
    t0 = time.time_ns()
    func()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.time_ns()
    result["latency_ms"] = (t2 - t0) / 1_000_000.0
    return result

def gen_result(m, run_result):
    r = E2EBenchmarkResult(device=m.device, device_num=m.device_num, test=m.test, num_examples=m.num_examples, batch_size=m.batch_size, result=dict())
    r.result["latency"] = run_result["latency_ms"] / 1000.0
    r.result["qps"] = r.num_examples / r.result["latency"]
    # add accuracy result if available
    if hasattr(m, "accuracy"):
        r.result["accuracy"] = m.accuracy
    return r

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", help="Full name of the end-to-end model.")
    parser.add_argument("-t", "--test", choices=["eval", "train"], default="eval", help="Which test to run.")
    parser.add_argument("--bs", type=int, help="Specify batch size.")
    args, extra_args = parser.parse_known_args()

    found = False
    Model = load_e2e_model_by_name(args.model)
    if not Model:
        print(f"Unable to find model matching {args.model}.")
        exit(-1)
    m = Model(test=args.test, batch_size=args.bs, extra_args=extra_args)
    test = getattr(m, args.test)
    result = gen_result(m, run(test))
    result_json = json.dumps(asdict(result))
    print(result_json)
