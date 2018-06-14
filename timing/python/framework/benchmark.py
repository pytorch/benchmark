import numpy as np
import csv
import random
import gc
import utils as bench_utils


class AttrDict(dict):
    def __repr__(self):
        keys = sorted(self.keys())
        result = ", ".join(k + "=" + str(self[k]) for k in keys)
        return "(" + result + ")"

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


class BenchmarkResult(dict):
    """
    Initialize with a double of seconds
    """

    def __init__(self, time, cpu, num_iter):
        self.time = float(time)
        self.cpu = float(cpu)
        self.num_iter = num_iter

    def get_time(self, unit="us"):
        return self.time * 1e6 / self.num_iter

    def get_cpu(self, unit="us"):
        return self.cpu * 1e6 / self.num_iter

    def get_num_iter(self):
        """
        Number of iterations run to get timings
        """
        return self.num_iter

    def __repr__(self):
        return ", ".join(k + "=" + str(v) for k, v in self.items())


class BenchmarkResults(object):
    def __init__(self):
        self.results = []

    def time_mean(self, unit="us"):
        return np.mean(list(map(lambda x: x.get_time(unit), self.results)))

    def time_std(self, unit="us"):
        return np.std(list(map(lambda x: x.get_time(unit), self.results)))

    def cpu_mean(self, unit="us"):
        return np.mean(list(map(lambda x: x.get_cpu(unit), self.results)))

    def cpu_std(self, unit="us"):
        return np.std(list(map(lambda x: x.get_cpu(unit), self.results)))

    def iter_mean(self):
        return np.mean(list(map(lambda x: x.get_num_iter(), self.results)))

    def append(self, result):
        assert isinstance(result, BenchmarkResult)
        self.results.append(result)


class Benchmark(object):
    def __init__(self):
        if "args" not in dir(self):
            import logging

            logger = logging.getLogger()
            logger.warn(
                "Benchmark " + str(self.__class__) + " has no args set."
                " Using empty dictionary."
            )
            self.args = [{}]
        if "user_counters" not in dir(self):
            import logging

            logger = logging.getLogger()
            logger.info(
                "Benchmark "
                + str(self.__class__)
                + " has no user counters set."
                " Using empty dictionary."
            )
            self.user_counters = {}
        cpus = bench_utils.get_cpu_list()
        for cpu in cpus:
            bench_utils.check_cpu_governor(cpu)
        self.state = AttrDict()


class ListBenchmark(Benchmark):
    """
    Basic benchmark class. Expects list of arguments and processes one at a
    time.
    """

    def __init__(self):
        super(ListBenchmark, self).__init__()
        if not isinstance(self.args, list):
            raise TypeError("args needs to be a list of arguments")


class GridBenchmark(Benchmark):
    """
    Creates a grid of arguments and calls benchmark with each.
    The arguments must be primited types such as strings or
    numbers and will be shallow(!) copied as part of the setup
    """

    def __init__(self):
        super(GridBenchmark, self).__init__()
        if not isinstance(self.args, dict):
            raise TypeError("args needs to be a dict of arguments")
        self.args = bench_utils.grid(self.args)


# TODO
# TODO: Implement regex filter for benchmark args
#           [--benchmark_filter=<regex>]
# DONE
#           [--benchmark_format=<console|json|csv>] No need for choice for now
#           [--benchmark_out_format=<json|console|csv>] No need for choice for now
#           [--benchmark_counters_tabular={true|false}] Done by default
#           [--v=<verbosity>]
#           [--benchmark_list_tests={true|false}]
#           [--benchmark_out=<filename>]
#           [--benchmark_min_time=<min_time>]
#           [--benchmark_repetitions=<num_repetitions>]
#           [--benchmark_report_aggregates_only={true|false}

# TODO: Write general setup script to check for environment setup
# TODO: Add additional checks to prevent user errors, e.g. wrong class methods


def run_func_benchmark(func, arg, state, settings):
    num_iter = 0
    start = bench_utils.timer()
    end = start
    cpu_start = bench_utils.cpu_timer()
    while end - start <= settings.benchmark_min_time:
        func(state, AttrDict(arg))
        end = bench_utils.timer()
        num_iter += 1
    result = BenchmarkResult(
        end - start, bench_utils.cpu_timer() - cpu_start, num_iter
    )
    return result


def make_print_row(row, row_format, header):
    status_str = ""

    special_fields = [
        "time_mean",
        "cpu_mean",
        "time_std",
        "cpu_std",
        "iter_mean",
    ]

    def process_header(header):
        v = row[header]
        if header in special_fields:
            v = int(v)
        return row_format[header].format(str(v))

    for i in range(len(header)):
        if header[i] in row:
            status_str += process_header(header[i])

    return status_str


def make_pretty_print_row_format(obj, header, header_labels, header_init):
    max_name_lens = {}
    for i in range(len(header)):
        max_name_lens[header[i]] = header_init[i] + 3

    def process_dict(d, header, header_labels):
        for k, v in d.items():
            if k not in max_name_lens:
                max_name_lens[k] = len(str(k)) + 3
                header += [k]
                header_labels += [k]
            max_name_lens[k] = max(max_name_lens[k], len(str(v)) + 3)

    for arg in obj.args:
        process_dict(arg, header, header_labels)
    process_dict(obj.user_counters, header, header_labels)
    row_format = {}
    for i in range(len(header)):
        row_format[header[i]] = "{:>" + str(max_name_lens[header[i]]) + "}"
    return row_format


def append_row(rows, row):
    for k, v in row.items():
        rows[k].append(v)


def get_all_jobs(obj, shuffle=False):
    jobs = []
    i = 1
    for func in dir(obj):
        if func.startswith("benchmark"):
            for arg in obj.args:
                config = AttrDict()
                config.number = i
                config.func = func
                config.arg = arg.copy()
                jobs.append(config)
                i += 1
    if shuffle:
        random.shuffle(jobs)
    return jobs


def init_row(job, obj, settings, name):
    row = job.arg.copy()
    row["benchmark"] = name
    row["repetitions"] = settings.benchmark_repetitions
    if settings.benchmark_warmup_repetitions > 0:
        row["warmup_repetitions"] = settings.benchmark_warmup_repetitions
    row["time_mean"] = 0
    row["time_std"] = 0
    row["cpu_mean"] = 0
    row["cpu_std"] = 0
    row["iter_mean"] = 0
    for counter, value in obj.user_counters.items():
        row[counter] = value
    return row


def run_benchmark_job(row, job, obj, settings):
    arg = job.arg
    func = getattr(obj, job.func)
    row["repetitions"] = settings.benchmark_repetitions
    results = BenchmarkResults()
    if "setupRun" in dir(obj):
        obj.setupRun(obj.state, AttrDict(arg))
    for i in range(
        settings.benchmark_repetitions + settings.benchmark_warmup_repetitions
    ):
        gc.collect()
        gc.collect()
        if "setup" in dir(obj):
            obj.setup(obj.state, AttrDict(arg))
        gc.collect()
        gc.collect()
        if i >= settings.benchmark_warmup_repetitions:
            results.append(run_func_benchmark(func, arg, obj.state, settings))
        gc.collect()
        gc.collect()
        if "teardown" in dir(obj):
            obj.teardown(obj.state, AttrDict(arg))
        gc.collect()
        gc.collect()
    if "teardownRun" in dir(obj):
        obj.teardownRun(obj.state, AttrDict(arg))
    for k, _ in obj.user_counters.items():
        if k in obj.state:
            row[k] = str(obj.state[k])
    row["time_mean"] = results.time_mean()
    row["time_std"] = results.time_std()
    row["cpu_mean"] = results.cpu_mean()
    row["cpu_std"] = results.cpu_std()
    row["iter_mean"] = results.iter_mean()
    return row


def calculate_progress(job_number, max_job_number, time_elapsed, info_format):
    avg_time = (float(time_elapsed)) / (job_number)
    time_left = int((max_job_number - job_number) * avg_time)
    info = info_format.format(
        "{}/{}".format(job_number, max_job_number),
        int(time_left / 60),
        "{:>02d} ".format(
            time_left % 60
        ),  # TODO: Fix misalignment - format string shouldn't need extra space
    )
    return info


def run_benchmark(obj, name, settings):
    """
    Create benchmark table. All times are in microseconds.
    """

    jobs = get_all_jobs(obj, settings.benchmark_shuffle)
    if len(jobs) == 0:
        return

    header = [
        "benchmark",
        "time_mean",
        "time_std",
        "cpu_mean",
        "cpu_std",
        "iter_mean",
        "repetitions",
        "warmup_repetitions",
    ]
    header_label = [
        "Benchmark",
        "Time mean (us)",
        "Time std (us)",
        "CPU mean (us)",
        "CPU std (us)",
        "Iter. mean",
        "Rep.",
        "Warmup Rep.",
    ]
    header_init = [max(12, len(name) + 3), 18, 18, 18, 18, 13, 14, 14]

    row_format = make_pretty_print_row_format(
        obj, header, header_label, header_init
    )
    rows = {}
    for head in header:
        rows[head] = []

    info_format = "{:>15}{:>10}:{:>2}"
    hstr = info_format.format("Job number", "ETA (mm", "ss)")
    row = init_row(jobs[0], obj, settings, name)
    for i in range(len(header)):
        if header[i] in row:
            hstr += row_format[header[i]].format(str(header_label[i]))
    print(len(hstr) * "-")
    print(hstr)
    print(len(hstr) * "-")
    out_csv_obj = None
    if settings.benchmark_out:
        out_csv_fd = open(settings.benchmark_out, "w")
        out_csv_obj = csv.DictWriter(out_csv_fd, header)
        out_csv_obj.writeheader()
    total_time = bench_utils.timer()
    for job in jobs:
        row = init_row(job, obj, settings, name)
        if not settings.dry_run:
            row = run_benchmark_job(row, job, obj, settings)
            append_row(rows, row)
            if out_csv_obj:
                out_csv_obj.writerow(row)
                out_csv_fd.flush()
        info = calculate_progress(
            job.number,
            len(jobs),
            bench_utils.timer() - total_time,
            info_format,
        )
        if settings.dry_run:
            print(info + make_print_row(row, row_format, header))
        else:
            print(info + make_print_row(row, row_format, header))


def create_benchmark_object(benchmark_class):
    try:
        assert issubclass(benchmark_class, Benchmark)
    except TypeError as e:
        raise TypeError(
            str(benchmark_class) + " must be subclass of Benchmark"
        )
    return benchmark_class()
