import importlib.util
import os
from pathlib import Path
import logging
import sys
import types
import unittest
from unittest import mock


class _FakeGroupMember:
    NON_GROUP_MEMBER = object()


class _FakeProcessGroupNCCL:
    class Options:
        def __init__(self):
            self.config = types.SimpleNamespace()


class TorchAxisCreationTest(unittest.TestCase):
    def setUp(self):
        self.new_group_calls = []

        fake_torch = types.ModuleType("torch")
        fake_torch.device = lambda device_type, index=None: ("cuda", int(device_type.split(":")[1])) if ":" in device_type else (device_type, index)
        fake_torch.set_default_device = mock.Mock()
        fake_torch.cuda = types.SimpleNamespace(
            current_device=lambda: 3,
            device_count=lambda: 4,
            set_device=mock.Mock(),
        )

        fake_dist = types.ModuleType("torch.distributed")
        fake_dist.group = types.SimpleNamespace(WORLD=object())
        fake_dist.GroupMember = _FakeGroupMember
        fake_dist.ProcessGroupNCCL = _FakeProcessGroupNCCL
        fake_dist.init_process_group = mock.Mock()
        fake_dist.get_rank = lambda: 0
        fake_dist.get_world_size = lambda: 2
        fake_dist.barrier = mock.Mock()
        fake_dist.destroy_process_group = mock.Mock()

        def new_group(**kwargs):
            self.new_group_calls.append(kwargs)
            return object()

        fake_dist.new_group = new_group
        fake_torch.distributed = fake_dist

        fake_async_work = types.ModuleType("maestro.core.utils.async_work")
        fake_async_work.AsyncWorkHandle = object
        fake_async_work.AsyncWorkHandleType = types.SimpleNamespace(TORCH="torch")
        fake_logging = types.ModuleType("maestro.core.utils.logging")
        fake_logging.get_logger = logging.getLogger

        self.modules = mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "torch.distributed": fake_dist,
                "maestro.core.utils.async_work": fake_async_work,
                "maestro.core.utils.logging": fake_logging,
            },
        )
        self.environment = mock.patch.dict(os.environ, {"LOCAL_RANK": "3"}, clear=True)
        self.environment.start()
        self.modules.start()
        spec = importlib.util.spec_from_file_location("maestro.core.utils.distributed", Path(__file__).resolve().parents[1] / "maestro/core/utils/distributed.py")
        self.distributed = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.distributed)
        self.new_group_calls.clear()

    def tearDown(self):
        self.modules.stop()
        self.environment.stop()

    def test_nccl_axis_group_is_eagerly_bound_to_current_device(self):
        group = self.distributed.nccl_torch_dist_utils.create_axis([[0, 1]])

        self.assertIsNotNone(group)
        self.assertEqual(
            self.new_group_calls,
            [
                {
                    "ranks": [0, 1],
                    "backend": "nccl",
                    "pg_options": None,
                    "device_id": ("cuda", 3),
                }
            ],
        )

    def test_traffic_class_is_set_on_nccl_group_options(self):
        for traffic_class in (0, 160):
            with self.subTest(traffic_class=traffic_class):
                self.distributed.nccl_torch_dist_utils.create_axis([[0, 1]], params={"traffic_class": traffic_class})
                options = self.new_group_calls[-1]["pg_options"]
                self.assertEqual(options.config.traffic_class, traffic_class)

    def test_non_nccl_axis_group_does_not_receive_cuda_device(self):
        dist_utils = self.distributed._TorchDistUtils.__new__(
            self.distributed._TorchDistUtils
        )
        dist_utils.backend = "gloo"

        group = dist_utils.create_axis([[0, 1]])

        self.assertIsNotNone(group)
        self.assertEqual(
            self.new_group_calls,
            [{"ranks": [0, 1], "backend": "gloo", "pg_options": None}],
        )


if __name__ == "__main__":
    unittest.main()
