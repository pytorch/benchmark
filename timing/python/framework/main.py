import argparse
import logging
import utils as bench_utils
import re
import random
from benchmark import (
    create_benchmark_object,
    run_benchmark_job,
    get_all_jobs,
    BenchmarkLogger,
    filter_jobs,
)


def main(argv, classes):
    benchmark_choices = []
    for member in classes:
        benchmark_choices.append(member.__name__)
    parser = argparse.ArgumentParser(
        description="Run benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--include",
        help="Only run the specified benchmarks",
        choices=benchmark_choices,
        default="all",
    )
    parser.add_argument(
        "--exclude",
        help="Don't run the specified benchmarks",
        choices=benchmark_choices,
        default="all",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all benchmarks and their arguments",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=30,
        choices=[0, 10, 20, 30, 40, 50],
        help=(
            "Threshold on logging module events to ignore."
            "Lower values lead to more verbose output"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do a dry run without collecting information",
    )
    parser.add_argument(
        "--benchmark-format",
        default="console",
        choices=["console", "csv"],
        help="Choose output format for stdout",
    )
    parser.add_argument(
        "--benchmark-filter", help="Run benchmarks which match specified regex"
    )
    parser.add_argument(
        "--benchmark-shuffle",
        action="store_true",
        help="Shuffle all benchmark jobs before executing",
    )
    parser.add_argument(
        "--benchmark-min-time",
        help="Min time per benchmark",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--benchmark-warmup-repetitions",
        help="Number of reptitions to ignore to warmup",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--benchmark-repetitions", help="Repeat benchmark", default=1, type=int
    )
    parser.add_argument(
        "--benchmark-out", help="Write benchmark results to file", default=None
    )
    parser.add_argument(
        "--benchmark-out-append",
        help="Append if file exists",
        action="store_true",
    )
    argv = argv[1:]
    args = parser.parse_args(argv)

    logging.basicConfig(level=int(args.verbose))
    logger = logging.getLogger()
    logger.info(bench_utils.show_cpu_info())
    for member in classes:
        if args.include is not "all":
            if member.__name__ not in args.include:
                continue
        if args.exclude is not "all":
            if member.__name__ in args.exclude:
                continue
        obj = create_benchmark_object(member)
        jobs = get_all_jobs(obj)
        if args.benchmark_filter:
            benchmark_filter = re.compile(args.benchmark_filter)
            jobs = filter_jobs(jobs, benchmark_filter)
        if args.benchmark_shuffle:
            random.shuffle(jobs)
        if args.list:
            [print(job) for job in jobs]
        else:
            blogger = BenchmarkLogger(
                args, obj, len(jobs), format=args.benchmark_format
            )
            if args.benchmark_out:
                if args.benchmark_out_append:
                    out_fd = open(args.benchmark_out, "a")
                else:
                    out_fd = open(args.benchmark_out, "w")
                bloggerout = BenchmarkLogger(
                    args, obj, len(jobs), format="csv", out_fd=out_fd
                )
            blogger.print_header()
            if args.benchmark_out:
                bloggerout.print_header()
            for job in jobs:
                row = run_benchmark_job(job, obj, args)
                blogger.print_row(row, job)
                if args.benchmark_out:
                    bloggerout.print_row(row, job)
            blogger.close()
            if args.benchmark_out:
                bloggerout.close()
                out_fd.close()
