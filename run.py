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

from torchbenchmark import load_model_by_name
import torch

WARMUP_ROUNDS = 3

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

        for s in streamlist:
            with torch.cuda.stream(s):
                func()

        end_event.record()
        torch.cuda.synchronize()

        print(f"Cuda StreamCount:{len(streamlist)}")
        print('{:<20} {:>20}'.format("GPU Time:", "%.3f milliseconds" % start_event.elapsed_time(end_event)), sep='')


def run_one_step(func, nwarmup=WARMUP_ROUNDS):
    # Warm-up `nwarmup` rounds
    for _i in range(nwarmup):
        func()

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
        t0 = time.time_ns()
        func()
        t1 = time.time_ns()
        print('{:<20} {:>20}'.format("CPU Total Wall Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')


def profile_one_step(func, nwarmup=WARMUP_ROUNDS):
    activity_groups = []
    if ((not args.profile_devices and args.device == 'cuda') or
            (args.profile_devices and 'cuda' in args.profile_devices)):
        print("Collecting CUDA activity.")
        activity_groups.append(profiler.ProfilerActivity.CUDA)

    if ((not args.profile_devices and args.device == 'cpu') or
            (args.profile_devices and 'cpu' in args.profile_devices)):
        print("Collecting CPU activity.")
        activity_groups.append(profiler.ProfilerActivity.CPU)

    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=nwarmup, active=1),
        activities=activity_groups,
        record_shapes=args.profile_detailed,
        profile_memory=args.profile_detailed,
        with_stack=args.profile_detailed,
        with_flops=args.profile_detailed,
        on_trace_ready=profiler.tensorboard_trace_handler(args.profile_folder)
    ) as prof:
        for _i in range(nwarmup + 1):
            func()
            torch.cuda.synchronize()  # Need to sync here to match run_one_step()'s timed run.
            prof.step()

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
    print(f"Saved TensorBoard Profiler traces to {args.profile_folder}.")


def _validate_devices(devices: str):
    devices_list = devices.split(",")
    valid_devices = ['cpu', 'cuda']
    for d in devices_list:
        if d not in valid_devices:
            raise ValueError(f'Invalid device {d} passed into --profile-devices. Expected devices: {valid_devices}.')
    return devices_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    SUPPORT_DEVICE_LIST = ["cpu", "cuda"]
    parser.add_argument("model", help="Full or partial name of a model to run.  If partial, picks the first match.")
    parser.add_argument("-d", "--device", choices=SUPPORT_DEVICE_LIST, default="cpu", help="Which device to use.")
    parser.add_argument("-m", "--mode", choices=["eager", "jit"], default="eager", help="Which mode to run.")
    parser.add_argument("-t", "--test", choices=["eval", "train"], default="eval", help="Which test to run.")
    parser.add_argument("--profile", action="store_true", help="Run the profiler around the function")
    parser.add_argument("--profile-folder", default="./logs", help="Save profiling model traces to this directory.")
    parser.add_argument("--profile-detailed", action="store_true",
                        help="Profiling includes record_shapes, profile_memory, with_stack, and with_flops.")
    parser.add_argument("--profile-devices", type=_validate_devices,
                        help="Profiling comma separated list of activities such as cpu,cuda.")
    parser.add_argument("--cudastreams", action="store_true",
                        help="Utilization test using increasing number of cuda streams.")
    parser.add_argument("--bs", type=int, help="Specify batch size to the test.")
    args = parser.parse_args()

    if args.cudastreams and not args.device == "cuda":
        print("cuda device required to use --cudastreams option!")
        exit(-1)

    found = False
    Model = load_model_by_name(args.model)
    if not Model:
        print(f"Unable to find model matching {args.model}.")
        exit(-1)

    print(f"Running {args.test} method from {Model.name} on {args.device} in {args.mode} mode.")

    # build the model and get the chosen test method
    if args.bs:
        try:
            if args.test == "eval":
                m = Model(device=args.device, jit=(args.mode == "jit"), eval_bs=args.bs)
            elif args.test == "train":
                m = Model(device=args.device, jit=(args.mode == "jit"), train_bs=args.bs)
        except:
            print(f"The model {args.model} doesn't support specifying batch size, please remove --bs argument in the commandline.")
            exit(1)
    else:
        m = Model(device=args.device, jit=(args.mode == "jit"))

    test = getattr(m, args.test)

    if args.profile:
        profile_one_step(test)
    elif args.cudastreams:
        run_one_step_with_cudastreams(test, 10)
    else:
        run_one_step(test)
