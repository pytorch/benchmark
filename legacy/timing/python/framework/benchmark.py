import numpy as np
import csv
import gc
import sys
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
                " Using empty list."
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


class BenchmarkLogger(object):
    """
    Provides estimate of time remaining based on time of creation and number
    of jobs associated with given object.
    """

    def __init__(
        self, settings, obj, num_jobs, format="console", out_fd=sys.stdout
    ):
        def make_pretty_print_row_format(
            obj, header, header_labels, header_init
        ):
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
                row_format[header[i]] = (
                    "{:>" + str(max_name_lens[header[i]]) + "}"
                )
            return row_format

        name = type(obj).__name__
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

        self.row_format = make_pretty_print_row_format(
            obj, header, header_label, header_init
        )
        self.obj = obj
        self.header = header
        self.header_label = header_label
        self.settings = settings
        self.info_format = "{:>15}{:>10}:{:>2}"

        self.num_jobs = num_jobs
        self.total_time = bench_utils.timer()
        self.out_fd = out_fd
        self.format = format

        if not (self.format == "console" or self.format == "csv"):
            raise TypeError("Wrong format. Must be console or csv")

        if self.format == "csv":
            self.csv_writer = csv.DictWriter(self.out_fd, header)

    def calculate_progress(self, job_number):
        time_elapsed = bench_utils.timer() - self.total_time
        max_job_number = self.num_jobs
        info_format = self.info_format
        # TODO: Fix misalignment - format string shouldn't need extra space
        avg_time = (float(time_elapsed)) / (job_number)
        time_left = int((max_job_number - job_number) * avg_time)
        info = info_format.format(
            "{}/{}".format(job_number, max_job_number),
            int(time_left / 60),
            "{:>02d} ".format(time_left % 60),
        )
        return info

    def print_header(self):
        obj = self.obj
        settings = self.settings
        header = self.header
        header_label = self.header_label
        row_format = self.row_format
        jobs = get_all_jobs(obj)
        info_format = self.info_format
        hstr = info_format.format("Job number", "ETA (mm", "ss)")
        row = init_row(jobs[0], obj, settings)
        for i in range(len(header)):
            if header[i] in row:
                hstr += row_format[header[i]].format(str(header_label[i]))
        if self.format == "console":
            self.out_fd.write(len(hstr) * "-" + "\n")
            self.out_fd.write(hstr + "\n")
            self.out_fd.write(len(hstr) * "-" + "\n")
            self.out_fd.flush()
        else:
            self.csv_writer.writeheader()

    def print_row(self, row, job):
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

        info = self.calculate_progress(job.number)
        if self.format == "console":
            self.out_fd.write(
                info + make_print_row(row, self.row_format, self.header) + "\n"
            )
            self.out_fd.flush()
        else:
            self.csv_writer.writerow(row)
            self.out_fd.flush()

    def close(self):
        pass


# TODO: Check if userdefined functions are proper (setup, setupRun, etc.)
# TODO: Implement regex filter for benchmark args
#           [--benchmark_filter=<regex>]
# DONE
#           [--benchmark_format=<console|json|csv>]
#           [--benchmark_out_format=<json|console|csv>] No need for now
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


def get_all_jobs(obj):
    jobs = []
    i = 1
    for func in dir(obj):
        if func.startswith("benchmark"):
            for arg in obj.args:
                config = AttrDict()
                config.number = i
                config.func = func
                config.arg = arg.copy()
                config.benchmark_name = type(obj).__name__
                jobs.append(config)
                i += 1
    return jobs


def filter_jobs(jobs_, benchmark_filter):
    jobs = []
    for job in jobs_:
        if benchmark_filter.match(str(job)):
            jobs.append(job)
    return jobs


def init_row(job, obj, settings):
    row = job.arg.copy()
    row["benchmark"] = type(obj).__name__
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


def run_benchmark_job(job, obj, settings):
    row = init_row(job, obj, settings)
    if settings.dry_run:
        return row
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
        result = run_func_benchmark(func, arg, obj.state, settings)
        if i >= settings.benchmark_warmup_repetitions:
            results.append(result)
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


def create_benchmark_object(benchmark_class):
    try:
        assert issubclass(benchmark_class, Benchmark)
    except TypeError as e:
        raise TypeError(
            str(benchmark_class) + " must be subclass of Benchmark"
        )
    return benchmark_class()
