"""
A lightweight runner that just sets up a model and runs one of its functions in a particular configuration.

Intended for debugging/exploration/profiling use cases, where the test/measurement harness is overhead.

DANGER: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
        has been installed.  This script intentionally does not automate or enforce setup steps.

Wall time provided for sanity but is not a sane benchmark measurement.
"""
import argparse
import time
import torch.profiler as profiler

from torchbenchmark import list_models
import torch
from torch.jit._fuser import fuser
import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()
import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.debug.metrics as metrics
from caffe2.python import workspace
# workspace.GlobalInit(['caffe2', '--caffe2_log_level=-4'])

WARMUP_ROUNDS = 8
MEASURE_ROUNDS = 8

def run_one_step(func, nwarmup=WARMUP_ROUNDS, nmeas=MEASURE_ROUNDS):
    # Warm-up `nwarmup` rounds
    for _i in range(nwarmup):
        out = func()

        if args.device == 'lazy':
            ltm.mark_step()

    if args.device == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Collect time_ns() instead of time() which does not provide better precision than 1
        # second according to https://docs.python.org/3/library/time.html#time.time.
        t0 = time.time_ns()
        func()
        t1 = time.time_ns()

        end_event.record()
        torch.cuda.synchronize()
        t2 = time.time_ns()

        # CPU Dispatch time include only the time it took to dispatch all the work to the GPU.
        # CPU Total Wall Time will include the CPU Dispatch, GPU time and device latencies.
        print('{:<20} {:>20}'.format("GPU Time:", "%.3f milliseconds" % start_event.elapsed_time(end_event)), sep='')
        print('{:<20} {:>20}'.format("CPU Dispatch Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')
        print('{:<20} {:>20}'.format("CPU Total Wall Time:", "%.3f milliseconds" % ((t2 - t0) / 1_000_000)), sep='')

    else:
        times = []
        for i in range(nmeas):
            t0 = time.time_ns()
            # we _have_ to keep the lazy tensor output alive outside of func, or we won't sync it!
            out = func(niter=1)

            # TODO(whc)
            # out.cuda() seems about as fast as mark_step() followed by wait_device_ops(), which is nice
            # but calling mark_step() by itself seems slower than expected
            out.cuda()
            # ltm.mark_step()
            # ltm.wait_device_ops()
            t1 = time.time_ns()
            times.append(((t1 - t0) / 1_000_000))
        print(f"CPU Total Wall Time (ms):\n[", ', '.join(["%.2f" % t for t in times]))


def _validate_devices(devices: str):
    devices_list = devices.split(",")
    valid_devices = ['cpu', 'cuda']
    for d in devices_list:
        if d not in valid_devices:
            raise ValueError(f'Invalid device {d} passed into --profile-devices. Expected devices: {valid_devices}.')
    return devices_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", help="Full or partial name of a model to run.  If partial, picks the first match.")
    parser.add_argument("-d", "--device", choices=["cpu", "cuda", "lazy"], default="cpu", help="Which device to use.")
    parser.add_argument("-t", "--test", choices=["eval", "train"], default="eval", help="Which test to run.")
    parser.add_argument("-f", "--fuser", choices=["fuser0", "fuser1", "fuser2"], default="fuser2", help="0=legacy, 1=nnc, 2=nvFuser")
    parser.add_argument("--bs", type=int, help="Specify batch size to the test.")
    args = parser.parse_args()

    found = False
    for Model in list_models():
        if args.model.lower() in Model.name.lower():
            found = True
            break
    if found:
        print(f"Running {args.test} method from {Model.name} on {args.device} using {args.fuser}.")
    else:
        print(f"Unable to find model matching {args.model}.")
        exit(-1)

    # build the model and get the chosen test method
    m = Model(device=args.device, jit=False)
    test = getattr(m, args.test)

    with fuser(args.fuser):
        run_one_step(test)

    if args.device == 'lazy':
        # print(metrics.counter_names())
        print("UncachedCompile ",  metrics.counter_value("UncachedCompile"))
        if "CachedCompile" in metrics.counter_names():
            print("CachedCompile ",  metrics.counter_value("CachedCompile"))
