import argparse
import time
import json
import os

warn_fields = [
    'node',  # Hostname
]
strict_check_fields = [
    'python_implementation_version',
    'processor',
    'release',  # OS kernel
    'system'  # OS family
]
def check_machine_info(old, new, ignore_mismatch=True):

    def warn(field):
        if old.get(field, "") != new.get(field, ""):
            print(f"Warning: machine info mismatch old[{field}]={old.get(field,'')}, new={new.get(field, '')}")

    def enforce(field):
        if old.get(field, "") != new.get(field, ""):
            raise RuntimeError(f"machine info mismatch old[{field}]={old.get(field,'')}, new={new.get(field, '')}")
    
    for field in warn_fields:
        warn(field)

    strict_check = warn if ignore_mismatch else enforce
    for field in strict_check_fields:
        strict_check(field)


def compare_benchmarks(old_benchmarks, new_benchmarks, tolerance=3.0):
    old_names = set([b['name'] for b in old_benchmarks])
    new_names = set([b['name'] for b in new_benchmarks])
    old_benchmarks = {b['name']: b for b in old_benchmarks}
    new_benchmarks = {b['name']: b for b in new_benchmarks}

    intersect = old_names.intersection(new_names)
    missing = old_names - new_names
    added = new_names - old_names

    for name in added:
        print(f"Found new benchmark: {name} with mean runtime {new_benchmarks[name]['stats']['mean']}")
    
    for name in missing:
        print(f"Missing data for benchmark {name}")

    regressions, improvements = [], []
    for name in intersect:
        old_std = old_benchmarks[name]['stats']['stddev']
        old_mean, new_mean = old_benchmarks[name]['stats']['mean'], new_benchmarks[name]['stats']['mean']
        diff_of_means = old_mean - new_mean
        if abs(diff_of_means) > tolerance * old_std:
            if diff_of_means < 0:
                print(f"Regression detected for {name}: old {old_mean}, new {new_mean}, diff/std {diff_of_means / old_std}")
                regressions.append(name)
            else:
                print(f"Improvement detected for {name}: old {old_mean}, new {new_mean}, diff/std {diff_of_means / old_std}")
                improvements.append(name)
        elif diff_of_means != 0:
            print(f"No change detected for {name}: old {old_mean}, new {new_mean}, diff/std {diff_of_means / old_std}")

    return len(regressions) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=argparse.FileType('r'),
                        help='old json')
    parser.add_argument("--new", required=True, type=argparse.FileType('r'),
                        help='old json')
    parser.add_argument("--ignore_machine_mismatch", action="store_true")
    parser.add_argument("--compare_tolerance", type=float, default=3.0,
                        help="how many standard deviations with respect to the "
                             "old mean to allow before flagging a change")

    args = parser.parse_args()
    old = json.load(args.old)
    new = json.load(args.new)
    
    check_machine_info(old['machine_info'], new['machine_info'], args.ignore_machine_mismatch)

    assert compare_benchmarks(old['benchmarks'], new['benchmarks'], tolerance=args.compare_tolerance), "Regressions Detected"