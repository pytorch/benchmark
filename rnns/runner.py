from __future__ import print_function
import inspect

from benchmarks.memnn import run_memnn


class AttrDict(dict):
    def __repr__(self):
        return ', '.join(k + '=' + str(v) for k, v in self.items())

    def __getattr__(self, name):
        return self[name]


class Benchmarks(object):
    def time_memnn(self):
        args = AttrDict({
            'warmup': 2,
            'benchmark': 2,
            'jit': False
        })
        # Should return a 'Bench' object
        return run_memnn(args)

    def time_memnn_jit(self):
        args = AttrDict({
            'warmup': 2,
            'benchmark': 18,
            'jit': True
        })
        return run_memnn(args)


def discover_benchmarks():
    benchmarks = Benchmarks()
    return inspect.getmembers(benchmarks, predicate=inspect.ismethod)


def main():
    timing_fns = discover_benchmarks()
    results = []
    for name, fn in timing_fns:
        try:
            print('{} {} {}'.format('-' * 30, name, '-' * 30))
            result = fn()
            results.append((name, result))
        except RuntimeError as err:
            print(err)
            results.append((name, None))

    print('{} {} {}'.format('-' * 30, 'summary', '-' * 30))
    for name, result in results:
        gpu_summary, cpu_summary = result.summary()
        print('{0: <16} {1:4.4f} msec gpu ({2:4.4f} msec cpu)'.format(
              name, gpu_summary.mean, cpu_summary.mean))


if __name__ == '__main__':
    main()
