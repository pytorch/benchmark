import argparse
from torchbenchmark.microbenchmarks import get_nvfuser_microbenchmarks


def run():
    parser = argparse.ArgumentParser(description="Run nvfuser microbenchmarks")
    parser.add_argument("--filter", nargs="*", default=[], help='List of benchmarks to test')
    parser.add_argument("--fusers", nargs="*", default=[], help='List of fusers to run tests on (options include "no_fuser", "fuser0", "fuser1", "fuser2")')
    args = parser.parse_args()

    microbenchmarks = get_nvfuser_microbenchmarks()
    if len(args.filter) > 0:
        microbenchmarks = [x for x in microbenchmarks if x.name in args.filter]
    if len(args.fusers) == 0:
        args.fusers = ["no_fuser", "fuser1", "fuser2"]

    for b in microbenchmarks:
        outputs = []
        for fuser in args.fusers:
            inputs = b.get_inputs()
            outputs.append((fuser, b.run_test(inputs, fuser)))
        print(f"{b.name}:", "; ".join(f"{name} = {time:.3f} ms" for name, time in outputs))

if __name__ == "__main__":
    run()
