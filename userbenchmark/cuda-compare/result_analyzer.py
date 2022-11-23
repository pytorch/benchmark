from pathlib import Path
import json
import re

def get_run(test_dir):
    run = {}
    testdir_name = test_dir.name
    regex = "cuda-(.*)-(.*)"
    g = re.match(regex, testdir_name).groups()
    run["test"] = g[0]
    run["cuda_version"] = g[1]
    eager_json = test_dir.joinpath("json", "eager.json")
    assert eager_json.exists(), f"Expected json path {str(eager_json)} doesn't exist."
    with open(eager_json, "r") as ej:
        run["result"] = json.load(ej)
    return run

def get_runs(work_dir: Path):
    runs = []
    for subdir in filter(lambda x: x.is_dir(), work_dir.iterdir()):
        run = get_run(subdir)
        runs.append(run)
    return runs

def add_test_results(runs, result_metrics, base_cuda_version):
    assert len(runs) >= 2, f"Expected more than 2 runs per group, getting {len(runs)}."
    base_run = list(filter(lambda x: x['cuda_version'] == base_cuda_version, runs))[0]
    for run in runs:
        if run["cuda_version"] == base_cuda_version:
            continue
        for test in run["result"]:
            test_name = f"{test['name']}-{test['test']}-{run['cuda_version']}-speedup"
            if test['status'] == 'OK':
                base_test = list(filter(lambda x: x['name'] == test['name'] and x['test'] == test['test'], base_run['result']))[0]
                result_metrics[test_name] = base_test['results']['latency_ms'] / test['results']['latency_ms']
            else:
                # status has error
                result_metrics[test_name] = "-1.0"
    return result_metrics

def analyze(result_dir):
    result_dir = Path(result_dir)
    assert result_dir.is_dir(), f"Expected directory {str(result_dir)} doesn't exist."
    result_metrics = { }
    runs = get_runs(result_dir)
    cuda_versions = sorted(map(lambda x: x["cuda_version"], runs))
    base_cuda_version = cuda_versions[0]
    cuda_train = list(filter(lambda x: x["test"] == "train", runs))
    add_test_results(cuda_train, result_metrics, base_cuda_version=base_cuda_version)
    cuda_eval = list(filter(lambda x: x["test"] == "eval", runs))
    add_test_results(cuda_eval, result_metrics, base_cuda_version=base_cuda_version)
    return result_metrics
