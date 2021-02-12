"""
Generate a fully specified benchmark configuration file, given a lightweight
specification and a complete source of benchmark data.

Specification File
------------------
Score hierarchy input intended to be as easy to construct as possible,
relying on automatic inference of unspecified weights, benchmark configs,
and normalization factors given a particular instance of benchmark data.

Structure:
    Root                             _
    - category                        |  required:
      - domain                        |  3 layers of organizational structure 
        - task                       _|

          - benchmark name           -   keyword match for root name in benchmark,
                                         omit children unless used
                                     _
            - train/eval              |  optional:
              - device                |  provide specific weights or 
                - compiler/runtime   _|  exclude particular configs by omission

Rules for describing the weight hierarchy
- everything is a dict, since at any level you could specify a weight
- if a weight is not specified, it is computed automatically with respect
its direct siblings.
- if specific benchmark configurations are omitted under a benchmark name,
all configurations present in the normalization data json are weighted equally

Normalization Data
------------------
Used to 'fill in the gaps' in the human written specification. 

- particular configurations (train/eval, device, compiler/runtime) present in 
this data are used to compute benchmark weights
- measurements from this data are used as normalization factors in score computation
  such that new data is scored relative to this data.

####
TODO
####
 - handle multiple normalization files, one for models, one for synthetic, etc
 - make explicit configuration choice for throughput vs runtime metrics
 - assert same machine used for all normalization files and freeze that in
"""
import argparse
import json
import yaml
from collections import defaultdict

# Support generate a config from benchmark data that runs partial of the spec
def generate_bench_cfg_partial(spec, norm, target):
    benchmark_names = [b['name'] for b in norm['benchmarks']]

    rec_defaultdict = lambda: defaultdict(rec_defaultdict)
    partial_spec = rec_defaultdict()

    def gen_partial_spec(category, domain, task, benchmark):
        found_benchmarks = [name for name in benchmark_names if benchmark in name]
        if len(found_benchmarks) > 0:
            partial_spec['hierarchy'][category][domain][task][benchmark] = None

    def visit_each_benchmark(spec, func):
        for category in spec['hierarchy']:
            category_spec = spec['hierarchy'][category]
            for domain in category_spec:
                tasks = category_spec[domain]
                for task in tasks:
                    benchmarks = tasks[task]
                    for benchmark in benchmarks:
                        func(category, domain, task, benchmark)

    visit_each_benchmark(spec, gen_partial_spec)
    return generate_bench_cfg(partial_spec, norm, target)

def check(spec):
    assert len(spec['hierarchy']) > 0, "Must specify at least one category"
    for category in spec['hierarchy']:
        category_spec = spec['hierarchy'][category]
        assert isinstance(category_spec, dict), f"Category {category} in spec must be non-empty"
        assert 'weight' not in category_spec, "TODO implement manual category weights"

        for domain in category_spec:
            tasks = category_spec[domain]
            assert isinstance(tasks, dict), f"Domain {category}:{domain} in spec must be non-empty"
            assert 'weight' not in tasks, "TODO implement manual domain weights"

            for task in tasks:
                benchmarks = tasks[task]
                assert isinstance(benchmarks, dict), f"Task {category}:{domain}:{task} in spec must be non-empty"
                assert 'weight' not in benchmarks, "TODO implement manual task weights"

                for benchmark in benchmarks:
                    assert benchmarks[benchmark] is None, "TODO handle benchmark as dict of config specs"
                    # assert 'weight' not in benchmarks[benchmark], "TODO implement manual benchmark weights"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", required=True,
        help="yaml file describing weight hierarchy")
    parser.add_argument("--normalization_data", required=True,
        help="pytest-benchmark json file used for generating normalization "
             "values and filling in unspecified benchmark configurations")
    parser.add_argument("--output_file", required=True,
        help="generated complete benchmark configuration")
    parser.add_argument("--target_score", default=1000,
        help="target score value given these normalizations and specifications")
    parser.add_argument("--partial",
                        action='store_true',
                        help="generates partial config if the benchmark only runs part of the spec."
                        "normally, the spec is supposed to define the set of benchmarks that's expected to exist,"
                        "and then the provided json data is expected to provide the norm values to match the spec."
                        "To simplify debugging, and not for normal score runs, we allow a convenience for producing"
                        "a score configuration that matches whatever json data is provided.")

    args = parser.parse_args()

    with open(args.specification) as spec_file:
        spec = yaml.full_load(spec_file)
    
    with open(args.normalization_data) as norm_file:
        norm = json.load(norm_file)

    with open(args.output_file, 'w') as out_file:
        check(spec)
        if args.partial:
            bench_cfg = generate_bench_cfg_partial(spec, norm, args.target_score)
        else:
            bench_cfg = generate_bench_cfg(spec, norm, args.target_score)
        yaml.dump(bench_cfg, out_file)
        
