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

def generate_bench_cfg(spec, norm, target):
    cfg = {
        'target': target,
        'benchmarks': {},
    }
    benchmark_names = [b['name'] for b in norm['benchmarks']]
    benchmark_norms = {b['name']: b['stats']['mean'] for b in norm['benchmarks']}

    assert len(spec['hierarchy']) > 0, "Must specify at least one category"
    category_weight = 1.0 / len(spec['hierarchy'])
    for category in spec['hierarchy']:
        
        category_spec = spec['hierarchy'][category]
        assert isinstance(category_spec, dict), f"Category {category} in spec must be non-empty"
        assert 'weight' not in category_spec, "TODO implement manual category weights"
        domain_weight = 1.0 / len(category_spec)
        
        for domain in category_spec:
            
            tasks = category_spec[domain]        
            assert isinstance(tasks, dict), f"Domain {category}:{domain} in spec must be non-empty"
            assert 'weight' not in tasks, "TODO implement manual domain weights"
            task_weight = 1.0 / len(tasks)
            
            for task in tasks:
                
                benchmarks = tasks[task]
                assert isinstance(benchmarks, dict), f"Task {category}:{domain}:{task} in spec must be non-empty"
                assert 'weight' not in benchmarks, "TODO implement manual task weights"
                benchmark_weight = 1.0 / len(benchmarks)
                
                for benchmark in benchmarks:
                    
                    assert benchmarks[benchmark] is None, "TODO handle benchmark as dict of config specs"
                    # assert 'weight' not in benchmarks[benchmark], "TODO implement manual benchmark weights"
                    found_benchmarks = [name for name in benchmark_names if benchmark in name]
                    assert len(found_benchmarks) > 0, f"No normalization data found for {benchmark}"
                    config_weight = 1.0 / len(found_benchmarks)
                    for b in found_benchmarks:
                        weight = domain_weight * task_weight * benchmark_weight * config_weight
                        cfg['benchmarks'][b] = {
                                'weight': weight,
                                'norm': benchmark_norms[b],
                        }
    return cfg

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
    args = parser.parse_args()

    with open(args.specification) as spec_file:
        spec = yaml.full_load(spec_file)
    
    with open(args.normalization_data) as norm_file:
        norm = json.load(norm_file)

    with open(args.output_file, 'w') as out_file:
        bench_cfg = generate_bench_cfg(spec, norm, args.target_score)
        yaml.dump(bench_cfg, out_file)
        
