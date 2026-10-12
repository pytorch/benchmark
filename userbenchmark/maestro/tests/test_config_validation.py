"""CPU regressions for validation before CUDA allocation/distributed imports."""
import ast
from copy import deepcopy
from decimal import Decimal
import re
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "maestro"


def load_definitions(path, names, namespace):
    tree = ast.parse((SRC / path).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Comm:
    @classmethod
    def _enrich_api_params_schema(cls, op):
        op.setdefault("vector_size", 24)


class Gemm:
    @classmethod
    def _enrich_api_params_schema(cls, op):
        for key in ("mat_a_shape", "mat_b_shape"):
            if key not in op:
                raise ValueError(f"Required parameter {key}")


class ConfigValidationTest(unittest.TestCase):
    def setUp(self):
        namespace = load_definitions("core/utils/format.py", {"parse_size"}, {"Union": __import__('typing').Union, "Number": __import__('numbers').Number, "Decimal": Decimal, "re": re})
        namespace.update(
            BlockRegistry=types.SimpleNamespace(get_block_cls=lambda name: Gemm if name == "torch_gemm" else Comm),
            CommBlock=Comm, GEMMBlock=Gemm, get_rank_and_world_size=lambda: (2, 4),
        )
        self.parse_pattern = load_definitions("core/config_parser.py", {"parse_pattern"}, namespace)["parse_pattern"]
        self.axes = {"test": [[0], [1, 2, 3]]}

    def parse(self, **op):
        return self.parse_pattern({"ops": [{"axis": "test", **op}]}, self.axes)["ops"][0]

    def test_gemm_dimensions_are_element_counts(self):
        result = self.parse(block="torch_gemm", mat_a_shape=["1K", 8], mat_b_shape=[8, 3])
        self.assertEqual(result["mat_a_shape"], (1024, 8))
        self.assertEqual(result["mat_b_shape"], (8, 3))

    def test_schema_defaults_are_applied_before_conversion(self):
        self.assertEqual(self.parse(block="torch_all_gather")["vector_size"], 12)

    def test_rank_local_divisibility(self):
        with self.assertRaisesRegex(ValueError, "team_size=3"):
            self.parse(block="torch_all_gather", vector_size=16)

    def test_invalid_inputs(self):
        for op in (
            {"block": "torch_all_gather", "vector_size": 3},
            {"block": "torch_all_gather", "vector_size": 0},
            {"block": "torch_gemm", "mat_a_shape": [2], "mat_b_shape": [2, 4]},
            {"block": "torch_gemm", "mat_a_shape": [2, 3], "mat_b_shape": [4, 2]},
            {"block": "torch_gemm", "mat_a_shape": [2, -1], "mat_b_shape": [-1, 2]},
            {"axis": "missing", "block": "torch_all_gather", "vector_size": 24},
        ):
            with self.subTest(op=op), self.assertRaises(ValueError):
                self.parse(**op)


class AxisParametersTest(unittest.TestCase):
    def setUp(self):
        namespace = {
            "Path": Path, "deepcopy": deepcopy, "Union": __import__('typing').Union,
            "_shared_timestamp": lambda: "test",
            "parse_axis_config": lambda config: [[0]],
            "parse_pattern": lambda pattern, axes: pattern,
            "parse_backends": lambda backends, timestamp: backends,
        }
        self.parse_config = load_definitions("core/config_parser.py", {"parse_config"}, namespace)["parse_config"]

    def test_axis_parameters_survive_config_parsing(self):
        for traffic_class in (0, 160):
            with self.subTest(traffic_class=traffic_class):
                config = self.parse_config({"axes": {"dp": {"teams": [[0]], "traffic_class": traffic_class}},
                                            "patterns": [{"ops": []}]})
                self.assertEqual(config["axis_params"], {"dp": {"traffic_class": traffic_class}})

    def test_invalid_traffic_class_is_rejected(self):
        for traffic_class in (-1, 256, True, "160", None):
            with self.subTest(traffic_class=traffic_class), self.assertRaisesRegex(ValueError, "traffic_class"):
                self.parse_config({"axes": {"dp": {"teams": [[0]], "traffic_class": traffic_class}},
                                   "patterns": [{"ops": []}]})


class ParseSizeTest(unittest.TestCase):
    def setUp(self):
        namespace = {"Union": __import__('typing').Union, "Number": __import__('numbers').Number, "Decimal": Decimal, "re": re}
        self.parse_size = load_definitions("core/utils/format.py", {"parse_size"}, namespace)["parse_size"]

    def test_legacy_and_explicit_binary_units(self):
        for value, expected in (("1K", 1024), ("2m", 2 * 1024**2), ("1G", 1024**3),
                                ("1KiB", 1024), ("1MiB", 1024**2), ("1gib", 1024**3), ("16B", 16)):
            with self.subTest(value=value):
                self.assertEqual(self.parse_size(value), expected)

    def test_fractional_magnitudes_resolve_to_integral_bytes(self):
        for value, expected in (("1.5M", 1572864), ("1.5MiB", 1572864), ("0.5K", 512),
                                ("1.25KiB", 1280), ("2.0", 2), (0, 0), (2.0, 2)):
            with self.subTest(value=value):
                self.assertEqual(self.parse_size(value), expected)

    def test_negative_nonintegral_and_invalid_sizes_are_rejected(self):
        for value in (-1, "-1", "-1.5M", 1.5, "1.5", "0.1K", True, "", "NaN", float("inf"), "1MB", "10000000000000000000000000000.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse_size(value)


class DistributedEnvironmentTest(unittest.TestCase):
    def test_torchrun_precedes_slurm(self):
        function = load_definitions("core/utils/distributed.py", {"get_rank_and_world_size"}, {"os": os})["get_rank_and_world_size"]
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "4", "SLURM_PROCID": "5", "SLURM_NTASKS": "8"}, clear=True):
            self.assertEqual(function(), (0, 4))

    def test_local_rank_and_single_visible_gpu(self):
        tree = ast.parse((SRC / "core/utils/distributed.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_TorchDistUtils")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_device")
        namespace = {"os": os, "torch": types.SimpleNamespace(device=str, cuda=types.SimpleNamespace(device_count=lambda: 4))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "get_device", "exec"), namespace)
        with patch.dict(os.environ, {"LOCAL_RANK": "2", "SLURM_LOCALID": "3"}, clear=True):
            self.assertEqual(namespace["get_device"](None), "cuda:2")
            namespace["torch"].cuda.device_count = lambda: 1
            self.assertEqual(namespace["get_device"](None), "cuda:0")


if __name__ == "__main__":
    unittest.main()
