import argparse
import logging
import utils as bench_utils
from benchmark import create_benchmark_object, run_benchmark, get_all_jobs


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
        help="Run only a subset of benchmarks",
        choices=benchmark_choices,
        nargs="?",
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
        "--benchmark-max-time",
        help="Max time per benchmark",
        default=60,
        type=int,
    )
    parser.add_argument(
        "--benchmark-warmup-repetitions",
        help="Number of reptitions to ignore to warmup",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--benchmark-min-iter",
        help="Min iterations per benchmark",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--benchmark-max-iter",
        help="Max iterations per benchmark",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--benchmark-repetitions", help="Repeat benchmark", default=1, type=int
    )
    parser.add_argument(
        "--benchmark-out", help="Write benchmark to file", default=None
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
        if args.list:
            obj = create_benchmark_object(member)
            jobs = get_all_jobs(obj, args.benchmark_shuffle)
            for job in jobs:
                print(job)
        else:
            obj = create_benchmark_object(member)
            logger.info("Running Benchmark " + str(member.__name__))
            run_benchmark(obj, member.__name__, args)
