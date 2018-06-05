import argparse
import logging
import utils as bench_utils
from benchmark import create_benchmark_object, run_benchmark


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

    logging.basicConfig(level=0)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(bench_utils.show_cpu_info())
    logger.info("Starting benchmarking")
    for member in classes:
        if args.include is not "all":
            if member.__name__ not in args.include:
                continue
        obj = create_benchmark_object(member)
        logging.info("Running Benchmark " + str(member.__name__))
        run_benchmark(obj, member.__name__, args)
