import argparse

from typing import List

def get_filters():
    pass

def get_experiments():
    pass

def _validate_tests(tests: str) -> List[str]:
    tests_list = list(map(lambda x: x.strip(), tests.split(",")))
    valid_tests = ['train', 'eval']
    for t in tests_list:
        if t not in valid_tests:
            raise ValueError(f'Invalid test {t} passed into --tests. Expected tests: {valid_tests}.')
    return tests_list

def _validate_devices(devices: str) -> List[str]:
    devices_list = list(map(lambda x: x.strip(), devices.split(",")))
    valid_devices = ['cpu', 'cuda']
    for d in devices_list:
        if d not in valid_devices:
            raise ValueError(f'Invalid device {d} passed into --devices. Expected devices: {valid_devices}.')
    return devices_list

def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, default=[],
                        help="Specify one or more models to run. If not set, trigger a sweep-run on all models.")
    parser.add_argument("-o", "--output", type=str, help="The default output json file.")
    args, unknown_args = parser.parse_known_args(args)
    return args, unknown_args