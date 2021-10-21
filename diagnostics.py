import gc
import itertools as it
import os
import pathlib
import shutil
import statistics
import unittest

import torch
from torchbenchmark import _list_model_paths, ModelTask


TRACE_FOLDER = os.path.join(pathlib.Path(__file__).parent.absolute(), "traces")
TIMEOUT = 900  # Seconds

_EXCLUDED_MODELS = {
    # This model uses both a custom train and custom eval loop.
    "nvidia_deeprecommender",

    # # TODO: eval times out and train fails if `niter` > 5.
    "speech_transformer",
}

_EXCLUDED_TEST_CASES = {
    # These test cases uses complex / custom training loops, so there is no
    # easy place to add `step_fn`.
    "test_pytorch_CycleGAN_and_pix2pix_cuda_train",
    "test_tts_angular_cuda_train",
    "test_yolov3_cuda_train",
}


class Collect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._results = {}

    @classmethod
    def tearDownClass(cls):
        if True:
            tests = sorted(cls._results.keys(), key=lambda s: s.lower())
            l = max(len(t) for t in tests)
            lines = [
                f"{'':>{l + 4}}                                         Profiler distortion (%)    ",
                f"{'':>{l + 4}}| ms / step   | Edge distortion (%) | (lightweight) :   (detailed) |"
            ]
            lines.append("-" * len(lines[-1]))
            for t in tests:
                result = cls._results[t]
                lines.append(
                    f"{t.ljust(l+4)}|  {result[0] * 100:>7.2f}{'':>4}|"
                    f"{result[1] * 100:>12.0f}{'':>9}|"
                    f"{result[2] * 100:>9.0f}{'':>6}:  {result[3] * 100:>9.0f}   |"
                )
            print("\n" + "\n".join(lines) + "\n")
        
        with open(os.path.join(TRACE_FOLDER, "summary.txt"), "wt") as f:
            f.write("\n".join(lines))

    def tearDown(self):
        gc.collect()

    def _test(self, *, model_path: str, test_name: str, trace_prefix: str, device: str, train: bool) -> None:
        task = ModelTask(model_path, timeout=TIMEOUT)
        baseline, steps, profiled_steps, profiled_steps_detailed = task.collect_diagnostics(
            trace_prefix=trace_prefix,
            device=device,
            train=train,
        )

        amortized_step_time = statistics.median([(t[-1] - t[0]) / (len(t) - 2) for t in baseline])
        median_step_time = statistics.median(steps)

        self._results[test_name] = (
            median_step_time,

            # Edge waste:
            #   The fraction of time spent in setup and teardown rather than
            #   executing the main step. (Ideally this should be zero.)
            max(0.0, amortized_step_time / median_step_time - 1.0),

            # Profiler overhead:
            #   How much distortion does running under the profiler cause?
            max(0.0, statistics.median(profiled_steps) / median_step_time - 1),

            # Profiler overhead: (detailed)
            #   What about when shape profiling, memory profiling, and stack
            #   collection are enabled?
            max(0.0, statistics.median(profiled_steps_detailed) / median_step_time - 1),
        )


def _add_test_case(*, model_path: str, device: str, train: bool) -> None:
    model_name = os.path.basename(model_path)
    test_name = f"test_{model_name}_{device}_{'train' if train else 'eval'}"

    if model_name in _EXCLUDED_MODELS or test_name in _EXCLUDED_TEST_CASES:
        def skip(self):
            self.skipTest("Test was manually excluded")
        setattr(Collect, test_name, skip)
        return

    os.makedirs(os.path.join(TRACE_FOLDER, model_name), exist_ok=True)
    trace_prefix = os.path.join(TRACE_FOLDER, model_name, f"{'train' if train else 'eval'}-{device}")

    def test(self):
        try:
            self._test(
                model_path=model_path,
                test_name=test_name,
                trace_prefix=trace_prefix,
                device=device,
                train=train,
            )

        except NotImplementedError:
            self.skipTest("Not implemented.")
    
    setattr(Collect, test_name, test)


def _populate() -> None:
    # TODO: We should collect CPU profiles, but it is currently very expensive.
    devices = []  # ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for model_path in _list_model_paths():
        for device, train in it.product(devices, [True, False]):
            _add_test_case(model_path=model_path, device=device, train=train)


if __name__ == "__main__":
    if os.path.exists(TRACE_FOLDER):
        shutil.rmtree(TRACE_FOLDER, ignore_errors=True)
    os.makedirs(TRACE_FOLDER)

    _populate()
    unittest.main()
