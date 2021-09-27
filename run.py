"""
A lightweight runner that just sets up a model and runs one of its functions in a particular configuration.

Intented for debugging/exploration/profiling use cases, where the test/measurement harness is overhead.

DANGER: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
        has been installed.  This script intentionally does not automate or enforce setup steps.

Wall time provided for sanity but is not a sane benchmark measurement.
"""
import argparse
import time
import torch.profiler as profiler

from torchbenchmark import list_models

import torch


def run_one_step_with_cudastreams(func, streamcount):

    print("Running Utilization Scaling Using Cuda Streams")

    streamlist = []
    for i in range(1, streamcount + 1, 1):

        # create additional streams and prime with load
        while len(streamlist) < i :
            s = torch.cuda.Stream()
            streamlist.append(s)

        for s in streamlist:
            with torch.cuda.stream(s):
                func()

        torch.cuda.synchronize()  # Wait for the events to be recorded!

        # now run benchmark using streams
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        t0 = time.time()
        for s in streamlist:
            with torch.cuda.stream(s):
                func()
        t1 = time.time()

        end_event.record()
        torch.cuda.synchronize()

        print(f"Cuda StreamCount:{len(streamlist)}: gpu time {start_event.elapsed_time(end_event)}")


def run_one_step(func):

    func()

    if args.device == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        t0 = time.time()
        func()
        t1 = time.time()

        end_event.record()
        torch.cuda.synchronize()
        print(f"Ran in {t1 - t0} seconds, gpu time {start_event.elapsed_time(end_event)}.")

    else:

        t0 = time.time()
        func()
        t1 = time.time()

        print(f"Ran in {t1 - t0} seconds.")


def profile_one_step(func, nwarmup=3):
    for i in range(nwarmup):
        func()

    activity_groups = []
    if args.device == "cuda":
        activity_groups.append(profiler.ProfilerActivity.CUDA)
    activity_groups.append(profiler.ProfilerActivity.CPU)

    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=0, active=1),
        activities=activity_groups,
        record_shapes=True,
        on_trace_ready=profiler.tensorboard_trace_handler(args.profile_folder)
    ) as prof:
        func()

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
    print(f"Saved TensorBoard Profiler traces to {args.profile_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", help="Full or partial name of a model to run.  If partial, picks the first match.")
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cpu", help="Which device to use.")
    parser.add_argument("-m", "--mode", choices=["eager", "jit"], default="eager", help="Which mode to run.")
    parser.add_argument("-t", "--test", choices=["eval", "train"], default="eval", help="Which test to run.")
    parser.add_argument("--profile", action="store_true", help="Run the profiler around the function")
    parser.add_argument("--profile-folder", default="./logs", help="Save profiling model traces to this directory.")
    parser.add_argument("--cudastreams", action="store_true", help="Utilization test using increasing number of cuda streams")
    args = parser.parse_args()

    if args.cudastreams and not args.device == "cuda":
        print("cuda device required to use --cudastreams option")
        exit(-1)

    found = False
    for Model in list_models():
        if args.model.lower() in Model.name.lower():
            found = True
            break
    if found:
        print(f"Running {args.test} method from {Model.name} on {args.device} in {args.mode} mode")
    else:
        print(f"Unable to find model matching {args.model}")
        exit(-1)

    # build the model and get the chosen test method
    m = Model(device=args.device, jit=(args.mode == "jit"))
    test = getattr(m, args.test)

    if args.profile:
        profile_one_step(test)
    elif args.cudastreams:
        run_one_step_with_cudastreams(test, 10)
    else:
        run_one_step(test)
