"""
A lightweight runner that just sets up a model and runs one of its functions in a particular configuration.

Intended for debugging/exploration/profiling use cases, where the test/measurement harness is overhead.

DANGER: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
        has been installed.  This script intentionally does not automate or enforce setup steps.

Wall time provided for sanity but is not a sane benchmark measurement.
"""
import argparse
import time
import numpy as np
import torch.profiler as profiler

from torchbenchmark import load_model_by_name
import torch

WARMUP_ROUNDS = 3
SUPPORT_DEVICE_LIST = ["cpu", "cuda"]
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    SUPPORT_DEVICE_LIST.append("mps")


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


def printResultSummaryTime(result_summary, metrics_needed=[], metrics_backend_mapping={}, model=None, model_analyzer=None):
    if args.device == "cuda":
        gpu_time = np.median(list(map(lambda x: x[0], result_summary)))
        cpu_walltime = np.median(list(map(lambda x: x[1], result_summary)))
        if hasattr(model, "NUM_BATCHES"):
            print('{:<20} {:>20}'.format("GPU Time per batch:", "%.3f milliseconds" %
                  (gpu_time / model.NUM_BATCHES), sep=''))
            print('{:<20} {:>20}'.format("CPU Wall Time per batch:", "%.3f milliseconds" %
                  (cpu_walltime / model.NUM_BATCHES), sep=''))
        else:
            print('{:<20} {:>20}'.format("GPU Time:", "%.3f milliseconds" % gpu_time, sep=''))
            print('{:<20} {:>20}'.format("CPU Total Wall Time:", "%.3f milliseconds" % cpu_walltime, sep=''))
    else:
        cpu_walltime = np.median(list(map(lambda x: x[0], result_summary)))
        print('{:<20} {:>20}'.format("CPU Total Wall Time:", "%.3f milliseconds" % cpu_walltime, sep=''))

    # if model_flops is not None, output the TFLOPs per sec
    if 'flops' in metrics_needed:
        if metrics_backend_mapping['flops'] == 'dcgm':
            tflops = model_analyzer.calculate_flops()
        else:
            flops, batch_size = m.get_flops()
            tflops = flops * batch_size / (cpu_walltime / 1.0e3) / 1.0e12
        print('{:<20} {:>20}'.format("FLOPS:", "%.4f TFLOPs per second" % tflops, sep=''))

    if 'gpu_peak_mem' in metrics_needed:
        gpu_peak_mem = model_analyzer.calculate_gpu_peak_mem()
        print('{:<20} {:>20}'.format("GPU Peak Memory:", "%.4f GB" % gpu_peak_mem, sep=''))
    if 'cpu_peak_mem' in metrics_needed:
        cpu_peak_mem = model_analyzer.calculate_cpu_peak_mem()
        print('{:<20} {:>20}'.format("CPU Peak Memory:", "%.4f GB" % cpu_peak_mem, sep=''))


def run_one_step(func, nwarmup=WARMUP_ROUNDS, num_iter=10, model=None, export_metrics_file=False, stress=0, metrics_needed=[], metrics_gpu_backend=None):
    # Warm-up `nwarmup` rounds
    for _i in range(nwarmup):
        func()

    result_summary = []
    metrics_backend_mapping = {}
    model_analyzer = None
    if metrics_needed:
        from components.model_analyzer.TorchBenchAnalyzer import ModelAnalyzer
        model_analyzer = ModelAnalyzer()
        if export_metrics_file:
            model_analyzer.set_export_csv_name(export_metrics_file)
        if 'gpu_peak_mem' in metrics_needed:
            model_analyzer.add_metric_gpu_peak_mem()
            metrics_backend_mapping['gpu_peak_mem'] = 'dcgm' if metrics_gpu_backend == 'dcgm' else 'nvml'
        if 'flops' in metrics_needed:
            if metrics_gpu_backend == 'dcgm':
                model_analyzer.add_metric_gpu_flops()
                metrics_backend_mapping['flops'] = 'dcgm'
            else:
                metrics_backend_mapping['flops'] = 'fvcore'
        if 'cpu_peak_mem' in metrics_needed:
            model_analyzer.add_metric_cpu_peak_mem()
        if metrics_gpu_backend == "default":
            model_analyzer.set_gpu_monitor_backend_nvml()
        for metric in metrics_backend_mapping:
            print(f"Metric {metric} is collected by {metrics_backend_mapping[metric]} backend")
        if 'cpu_peak_mem' in metrics_needed:
            print("Metric cpu_peak_mem is collected by psutil.Process.")
        model_analyzer.start_monitor()

    if stress:
        cur_time = time.time_ns()
        start_time = cur_time
        target_time = stress * 1e9 + start_time
        num_iter = -1
        last_time = start_time
    _i = 0
    last_it = 0
    first_print_out = True
    while (not stress and _i < num_iter) or (stress and cur_time < target_time) :
        if args.device == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Collect time_ns() instead of time() which does not provide better precision than 1
            # second according to https://docs.python.org/3/library/time.html#time.time.
            t0 = time.time_ns()
            start_event.record()
            func()
            end_event.record()
            torch.cuda.synchronize()
            t1 = time.time_ns()
            result_summary.append((start_event.elapsed_time(end_event), (t1 - t0) / 1_000_000))
        elif args.device == "mps":
            t0 = time.time_ns()
            func()
            t1 = time.time_ns()
            wall_latency = t1 - t0
            # TODO: modify this to add GPU time as well
            result_summary.append([(t1 - t0) / 1_000_000])
        else:
            t0 = time.time_ns()
            func()
            t1 = time.time_ns()
            result_summary.append([(t1 - t0) / 1_000_000])
        if stress:
            cur_time = time.time_ns()
            # print out the status every 10s.
            if (cur_time - last_time) >= 10 * 1e9:
                if first_print_out:
                    print('|{:^20}|{:^20}|{:^20}|'.format("Iterations", "Time/Iteration(ms)", "Rest Time(s)"))
                    first_print_out = False
                est = (target_time - cur_time) / 1e9
                time_per_it = (cur_time - last_time) / (_i - last_it) / 1e6
                print('|{:^20}|{:^20}|{:^20}|'.format("%d" % _i, "%.2f" % time_per_it , "%d" % int(est)))
                last_time = cur_time
                last_it = _i
        _i += 1

    if model_analyzer is not None:
        model_analyzer.stop_monitor()
        model_analyzer.aggregate()

    printResultSummaryTime(result_summary, metrics_needed, metrics_backend_mapping, model, model_analyzer)

    if export_metrics_file:
        model_analyzer.export_all_records_to_csv()


def profile_one_step(func, nwarmup=WARMUP_ROUNDS):
    activity_groups = []
    result_summary = []
    device_to_activity = {'cuda': profiler.ProfilerActivity.CUDA, 'cpu': profiler.ProfilerActivity.CPU}
    if args.profile_devices:
        activity_groups = [
            device_to_activity[device] for device in args.profile_devices if (device in device_to_activity)
        ]
    else:
        if args.device == 'cuda':
            activity_groups = [
                profiler.ProfilerActivity.CUDA,
                profiler.ProfilerActivity.CPU,
            ]
        elif args.device == 'cpu':
            activity_groups = [profiler.ProfilerActivity.CPU]

    if args.profile_eg:
        from datetime import datetime
        import os
        from torch.profiler import ExecutionGraphObserver
        start_time = datetime.now()
        timestamp = int(datetime.timestamp(start_time))
        eg_file = f"{args.model}_{timestamp}_eg.json"
        eg = ExecutionGraphObserver()
        if not os.path.exists(args.profile_eg_folder):
            os.makedirs(args.profile_eg_folder)
        eg.register_callback(f"{args.profile_eg_folder}/{eg_file}")
        nwarmup = 0
        eg.start()
    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=nwarmup, active=1, repeat=1),
        activities=activity_groups,
        record_shapes=args.profile_detailed,
        profile_memory=args.profile_detailed,
        with_stack=args.profile_detailed,
        with_flops=args.profile_detailed,
        on_trace_ready=profiler.tensorboard_trace_handler(args.profile_folder)
    ) as prof:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for i in range(nwarmup + 1):
            t0 = time.time_ns()
            start_event.record()
            func()
            torch.cuda.synchronize()  # Need to sync here to match run_one_step()'s timed run.
            end_event.record()
            t1 = time.time_ns()
            if i >= nwarmup:
                result_summary.append((start_event.elapsed_time(end_event), (t1 - t0) / 1_000_000))
            prof.step()
    if args.profile_eg and eg:
        eg.stop()
        eg.unregister_callback()
        print(f"Save Exeution Graph to : {args.profile_eg_folder}/{eg_file}")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
    print(f"Saved TensorBoard Profiler traces to {args.profile_folder}.")

    printResultSummaryTime(result_summary)

def _validate_devices(devices: str):
    devices_list = devices.split(",")
    valid_devices = SUPPORT_DEVICE_LIST
    for d in devices_list:
        if d not in valid_devices:
            raise ValueError(f'Invalid device {d} passed into --profile-devices. Expected devices: {valid_devices}.')
    return devices_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
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
    parser.add_argument("--profile-eg", action="store_true", help="Collect execution graph by PARAM")
    parser.add_argument("--profile-eg-folder", default="./eg_logs",
                        help="Save execution graph traces to this directory.")
    parser.add_argument("--cudastreams", action="store_true",
                        help="Utilization test using increasing number of cuda streams.")
    parser.add_argument("--bs", type=int, help="Specify batch size to the test.")
    parser.add_argument("--flops", choices=["fvcore", "dcgm"], help="Return the flops result.")
    parser.add_argument("--export-metrics", action="store_true",
                        help="Export all specified metrics records to a csv file. The default csv file name is [model_name]_all_metrics.csv.")
    parser.add_argument("--stress", type=float, default=0, help="Specify execution time (seconds) to stress devices.")
    parser.add_argument("--metrics", type=str,
                        help="Specify metrics [cpu_peak_mem,gpu_peak_mem,flops]to be collected. The metrics are separated by comma such as cpu_peak_mem,gpu_peak_mem.")
    parser.add_argument("--metrics-gpu-backend", choices=["dcgm", "default"], default="default", help="""Specify the backend [dcgm, default] to collect metrics. \nIn default mode, the latency(execution time) is collected by time.time_ns() and it is always enabled. Optionally, 
    \n  - you can specify cpu peak memory usage by --metrics cpu_peak_mem, and it is collected by psutil.Process().  \n  - you can specify gpu peak memory usage by --metrics gpu_peak_mem, and it is collected by nvml library.\n  - you can specify flops by --metrics flops, and it is collected by fvcore.\nIn dcgm mode, the latency(execution time) is collected by time.time_ns() and it is always enabled. Optionally,\n  - you can specify cpu peak memory usage by --metrics cpu_peak_mem, and it is collected by psutil.Process().\n  - you can specify cpu and gpu peak memory usage by --metrics cpu_peak_mem,gpu_peak_mem, and they are collected by dcgm library.""")
    args, extra_args = parser.parse_known_args()
    if args.cudastreams and not args.device == "cuda":
        print("cuda device required to use --cudastreams option!")
        exit(-1)

    found = False
    Model = load_model_by_name(args.model)
    if not Model:
        print(f"Unable to find model matching {args.model}.")
        exit(-1)

    m = Model(device=args.device, test=args.test, jit=(args.mode == "jit"), batch_size=args.bs, extra_args=extra_args)
    print(f"Running {args.test} method from {Model.name} on {args.device} in {args.mode} mode with input batch size {m.batch_size}.")

    test = m.invoke
    metrics_needed = [_ for _ in args.metrics.split(',') if _.strip()] if args.metrics else []
    metrics_gpu_backend = args.metrics_gpu_backend
    if metrics_needed:
        if metrics_gpu_backend == 'dcgm':
            from components.model_analyzer.TorchBenchAnalyzer import check_dcgm
            check_dcgm()
        elif 'gpu_peak_mem' in metrics_needed:
            from components.model_analyzer.TorchBenchAnalyzer import check_nvml
            check_nvml()
        if 'gpu_peak_mem' in metrics_needed or ('flops' in metrics_needed and metrics_gpu_backend == 'dcgm'):
            assert args.device == 'cuda', "gpu_peak_mem and flops:dcgm are only available for cuda device."
        if 'flops' in metrics_needed and metrics_gpu_backend == 'default':
            assert hasattr(m, "get_flops"), f"The model {args.model} does not support calculating flops."
            m.get_flops()
    if args.export_metrics:
        if not args.metrics:
            print("You have to specifiy at least one metrics to export.")
            exit(-1)
        export_metrics_file = "%s_all_metrics.csv" % args.model
    else:
        export_metrics_file = False
    if args.profile:
        profile_one_step(test)
    elif args.cudastreams:
        run_one_step_with_cudastreams(test, 10)
    else:
        run_one_step(test, model=m, export_metrics_file=export_metrics_file,
                     stress=args.stress, metrics_needed=metrics_needed, metrics_gpu_backend=args.metrics_gpu_backend)
    if hasattr(m, 'correctness'):
        print('{:<20} {:>20}'.format("Correctness: ", str(m.correctness)), sep='')
