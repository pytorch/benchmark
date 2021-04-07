"""
A lightweight runner that just sets up a model and runs one of its functions in a particular configuration.

Intented for debugging/exploration/profiling use cases, where the test/measurement harness is overhead.

DANGER: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
        has been installed.  This script intentionally does not automate or enforce setup steps.

Wall time provided for sanity but is not a sane benchmark measurement.
"""
import argparse
from torchbenchmark import list_models
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", help="Full or partial name of a model to run.  If partial, picks the first match.")
    parser.add_argument("-d", "--device", choices=["cpu",  "cuda"], default="cpu", help="Which device to use.")
    parser.add_argument("-m", "--mode", choices=["eager",  "jit"], default="eager", help="Which mode to run.")
    parser.add_argument("-t", "--test", choices=["eval",  "train"], default="eval", help="Which test to run.")
    args = parser.parse_args()

    found = False
    for Model in list_models():
        if args.model in Model.name.lower():
            found = True
            break
    if found:
        print(f"Running {args.test} method from {Model.name} on {args.device} in {args.mode} mode")
    else:
        print(f"Unable to find model matching {args.model}")
        exit(-1)

    # build the model and get the chosen test method 
    m = Model(args.device, args.mode)
    test = getattr(m, args.test)

    # run one step
    t0 = time.time()
    test()
    t1 = time.time()

    print(f"Ran in {t1 - t0} seconds.")


