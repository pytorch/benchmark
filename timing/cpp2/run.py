import json
import subprocess
from functools import partial
from io import StringIO

run_with_output = partial(subprocess.check_output)


def run_benchmark(name):
    # NB: assumes aten_overheads has already been built (see README.md)
    bench = run_with_output(['./build/' + name,
                             '--benchmark_format=json'])
    data = json.loads(bench.decode('utf-8'))
    # gbenchmark json output: benchmarks -> [name, cpu_time]
    output = {}
    for result in data['benchmarks']:
        output[result['name']] = result['cpu_time']
    return {name: output}


if __name__ == '__main__':
    benchmarks = [
        'aten_overheads',
        'tensor_properties',
        'tensor_allocation',
        'torch_empty',
        'tensor_shape'
    ]
    output = {}
    for benchmark in benchmarks:
        result = run_benchmark(benchmark)
        for k, v in result.items():
            output[k] = v
    print(output)
