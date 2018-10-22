import json
import subprocess
from functools import partial
from io import StringIO

run_with_output = partial(subprocess.check_output)


def run_benchmark():
    # NB: assumes aten_overheads has already been built (see README.md)
    bench = run_with_output(['./build/aten_overheads',
                             '--benchmark_format=json'])
    data = json.loads(bench.decode('utf-8'))
    # gbenchmark json output: benchmarks -> [name, cpu_time]
    output = {}
    for result in data['benchmarks']:
        output[result['name']] = result['cpu_time']
    print({'tensor_bench': output})


if __name__ == '__main__':
    run_benchmark()
