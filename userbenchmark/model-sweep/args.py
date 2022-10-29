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
    parser.add_argument("-m", "--models", nargs='+', default=[],
                        help="Specify one or more models to run. If not set, trigger a sweep-run on all models.")
    parser.add_argument("-t", "--tests", required=True, type=_validate_tests, help="Specify tests, choice of train, or eval.")
    parser.add_argument("-d", "--devices", required=True, type=_validate_devices, help="Specify devices, choice of cpu, or cuda.")
    parser.add_argument("-b", "--bs", type=int, help="Specify the batch size.")
    parser.add_argument("--jit", action='store_true', help="Turn on torchscript.")
    parser.add_argument("-o", "--output", type=str, help="The default output json file.")