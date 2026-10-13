import importlib.util
from pathlib import Path
import logging
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock


fake_distributed = types.ModuleType("maestro.core.utils.distributed")
fake_distributed.nccl_torch_dist_utils = None
fake_logging = types.ModuleType("maestro.core.utils.logging")
fake_logging.get_logger = logging.getLogger
fake_torch = types.ModuleType("torch")
fake_torch.distributed = types.SimpleNamespace(ProcessGroup=object)
fake_torch.cuda = types.SimpleNamespace(Event=object, Stream=object)
with mock.patch.dict(
    sys.modules,
    {
        "torch": fake_torch,
        "maestro.core.utils.distributed": fake_distributed,
        "maestro.core.utils.logging": fake_logging,
    },
):
    spec = importlib.util.spec_from_file_location("maestro.core.axis", Path(__file__).resolve().parents[1] / "maestro/core/axis.py")
    axis_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(axis_module)
    Axis, current_axis = axis_module.Axis, axis_module.current_axis


ENV_KEY = "MAESTRO_TEST_AXIS_ENV"


class _FakeDistUtils:
    def __init__(self):
        self.env_at_create = None

    def create_axis(self, groups, params=None):
        self.params_at_create = params
        self.env_at_create = os.environ.get(ENV_KEY)
        return object()

    @staticmethod
    def get_rank():
        return 0

    @staticmethod
    def get_world_size():
        return 1


class _FakeAxis(Axis):
    dist_utils = _FakeDistUtils()

    @contextmanager
    def _axis_ctx(self):
        yield

    @staticmethod
    def create_event():
        return None

    @staticmethod
    def record_event(event):
        pass

    def wait_event(self, event):
        pass

    def synchronize(self):
        pass

    @classmethod
    def synchronize_all(cls):
        pass


class AxisEnvironmentTest(unittest.TestCase):
    def test_default_ignores_overrides_and_warns(self):
        with mock.patch.dict(os.environ, {ENV_KEY: "process"}, clear=True):
            with self.assertLogs(axis_module.logger, level="WARNING") as logs:
                _FakeAxis([[0]], envs={ENV_KEY: "axis"})
            self.assertIn("Ignoring axis environment overrides", logs.output[0])
            self.assertEqual(_FakeAxis.dist_utils.env_at_create, "process")
            self.assertEqual(os.environ[ENV_KEY], "process")

    def test_explicit_false_ignores_overrides(self):
        for value in ("0", "false", "no", "off", "unexpected"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {ENV_KEY: "process", "MAESTRO_USE_PER_AXIS_ENV_VARS": value}, clear=True):
                with self.assertLogs(axis_module.logger, level="WARNING"):
                    _FakeAxis([[0]], envs={ENV_KEY: "axis"})
                self.assertEqual(_FakeAxis.dist_utils.env_at_create, "process")

    def test_opt_in_scopes_overrides_to_group_creation(self):
        for value in ("1", "TRUE", "yes", "On"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {ENV_KEY: "process", "MAESTRO_USE_PER_AXIS_ENV_VARS": value}, clear=True):
                axis = _FakeAxis([[0]], envs={ENV_KEY: "axis"})
                self.assertEqual(_FakeAxis.dist_utils.env_at_create, "axis")
                self.assertEqual(os.environ[ENV_KEY], "process")
                with axis.use_axis():
                    self.assertIs(current_axis.get(), axis)
                    self.assertEqual(os.environ[ENV_KEY], "process")
                self.assertIsNone(current_axis.get())

    def test_opt_in_restores_environment_after_failure(self):
        with mock.patch.dict(os.environ, {ENV_KEY: "process", "MAESTRO_USE_PER_AXIS_ENV_VARS": "1"}, clear=True):
            with self.assertRaises(RuntimeError):
                with axis_module._temporary_environ({ENV_KEY: "axis", "MAESTRO_NEW_ENV": "value"}):
                    self.assertEqual(os.environ[ENV_KEY], "axis")
                    raise RuntimeError("group creation failed")
            self.assertEqual(os.environ[ENV_KEY], "process")
            self.assertNotIn("MAESTRO_NEW_ENV", os.environ)

    def test_process_group_params_are_forwarded(self):
        for traffic_class in (0, 160):
            with self.subTest(traffic_class=traffic_class):
                _FakeAxis([[0]], params={"traffic_class": traffic_class})
                self.assertEqual(_FakeAxis.dist_utils.params_at_create, {"traffic_class": traffic_class})

    def test_rank_local_team_size(self):
        self.assertEqual(_FakeAxis([[1, 2], [0]]).team_size(), 1)

    def test_missing_membership_raises_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "Rank 0 is not in any group"):
            _FakeAxis([[1]])


if __name__ == "__main__":
    unittest.main()
