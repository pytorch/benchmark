"""CPU regression checks for storage ownership and CUPTI warmup boundaries."""
import importlib.util
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


class Tensor:
    def __init__(self, size, fill=0, storage=None, offset=0):
        self.storage = [fill] * size if storage is None else storage
        self.offset, self.size = offset, size
        self.dtype = types.SimpleNamespace(itemsize=2)
        self.shape = (size,)

    def __getitem__(self, region):
        start, stop, _ = region.indices(self.size)
        return Tensor(stop - start, storage=self.storage, offset=self.offset + start)

    def numel(self):
        return self.size

    def fill_(self, value):
        self.storage[self.offset:self.offset + self.size] = [value] * self.size
        return self

    def zero_(self):
        return self.fill_(0)

    def reshape(self, *shape):
        self.shape = shape
        return self


class BufferProfilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load production modules in an isolated namespace so CPU mocks do not
        # leak into the rest of the test suite or require a CUDA installation.
        prefix = '_maestro_buffer_tests'
        modules = {}
        def module(name, **attrs):
            result = types.ModuleType(name)
            result.__dict__.update(attrs)
            modules[name] = result
            return result
        for suffix in ('', '.core', '.core.utils', '.core.profilers', '.blocks'):
            module(prefix + suffix).__path__ = []
        module(prefix + '.core.utils.logging', get_logger=logging.getLogger,
               get_root_rank_logger=lambda: logging.getLogger('root'))
        module(prefix + '.core.utils.format', format_size=str)
        module(prefix + '.core.utils.distributed', get_rank_and_world_size=lambda: (0, 2),
               _DistUtils=object, root_print=lambda *a: None, dist_print=lambda *a: None)
        module(prefix + '.core.axis', Axis=object, TorchAxis=object)
        module(prefix + '.core.utils.start_gate', CUDAStartGate=object)
        cls.fake_torch = module('torch', Tensor=Tensor, dtype=object, float16='float16',
                                zeros=lambda n, **kw: Tensor(n), ones=lambda n, **kw: Tensor(n, 1))
        module('cupti')  # CUPTI absence must not prevent testing reset boundaries.
        cls.module_patch = mock.patch.dict(sys.modules, modules)
        cls.module_patch.start()
        source = Path(__file__).resolve().parents[1] / 'maestro' / 'core'
        def load(name, path):
            spec = importlib.util.spec_from_file_location(prefix + '.core.' + name, source / path)
            result = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = result
            spec.loader.exec_module(result)
            return result
        cls.base = load('profilers.base', 'profilers/base.py')
        modules[prefix + '.core.profilers'].profiler_ctx = cls.base.profiler_ctx
        cls.block = load('block', 'block.py')
        cls.orchestrator = load('orchestrator', 'orchestrator.py')
        cls.cupti = load('profilers.cupti', 'profilers/cupti.py')

    @classmethod
    def tearDownClass(cls):
        cls.module_patch.stop()

    def make_comm_cls(self):
        return type('TestComm', (self.block.CommBlock,), {'run': lambda self: None})

    def test_collective_buffers_do_not_alias_and_reject_small_cache(self):
        comm = self.make_comm_cls()
        axis = types.SimpleNamespace(team_size=lambda: 2)
        result = comm.from_api_params(axis, 'comm', {'vector_size': 8}, Tensor(16))
        result.src_buf.fill_(7)
        self.assertEqual(result.dst_buf.storage[8:], [0] * 8)
        with self.assertRaisesRegex(ValueError, 'needs 16 elements'):
            comm.from_api_params(axis, 'comm', {'vector_size': 8}, Tensor(8))

    def test_pool_sums_concurrent_blocks_and_reuses_patterns(self):
        comm = self.make_comm_cls()
        obj = object.__new__(self.orchestrator.Orchestrator)
        obj.axes = {'x': types.SimpleNamespace(team_size=lambda: 2)}
        obj.patterns = [{'ops': [{'axis': 'x', 'block': 'test', 'vector_size': 8}] * 2},
                        {'ops': [{'axis': 'x', 'block': 'test', 'vector_size': 4}]}]
        obj.register_buffers = False
        obj._tensors_pool = {}
        obj.axis_cls = types.SimpleNamespace(synchronize_all=mock.Mock())
        with mock.patch.object(self.block.BlockRegistry, 'get_block_cls', return_value=comm):
            obj._prepare_tensors_pool()
            self.assertEqual(obj._tensors_pool['main'].numel(), 32)
            first, second = obj._create_blocks(obj.patterns[0])
        first.dst_buf.fill_(9)
        self.assertEqual(second.src_buf.storage[16:], [0] * 16)
        obj.axis_cls.synchronize_all.assert_called_once()

    def test_fallback_allocations_use_fp16_and_axis_sizes(self):
        comm = self.make_comm_cls()
        axis = types.SimpleNamespace(team_size=lambda: 2)
        with mock.patch.object(self.fake_torch, 'zeros', wraps=self.fake_torch.zeros) as allocate:
            comm.from_api_params(axis, 'comm', {'vector_size': 8})
        allocate.assert_called_once_with(16, device='cuda', dtype='float16')
        gemm = type('TestGemm', (self.block.GEMMBlock,), {'run': lambda self: None})
        with mock.patch.object(self.fake_torch, 'ones', wraps=self.fake_torch.ones) as allocate:
            result = gemm.from_api_params(axis, 'gemm', {'mat_a_shape': [2, 3], 'mat_b_shape': [3, 4]})
        allocate.assert_called_once_with(18, device='cuda', dtype='float16')
        self.assertEqual(result.mat_a.shape, (2, 3))
        self.assertEqual(result.mat_b.offset, 6)

    def test_active_reset_flushes_warmup_before_clearing(self):
        obj = object.__new__(self.cupti.CuptiProfiler)
        obj._active = True
        obj._results = [{'warmup': 1}]
        def flush(_):
            obj._results.append({'pending_warmup': 1})
        api = types.SimpleNamespace(activity_flush_all=mock.Mock(side_effect=flush))
        with mock.patch.object(self.cupti, 'cupti', api, create=True):
            obj.reset()
            self.assertEqual(obj.get_results(), [])
            obj._active = False
            obj.reset()
        api.activity_flush_all.assert_called_once_with(1)

    def test_extra_activities_rejected_collectively_before_reporting(self):
        obj = object.__new__(self.orchestrator.Orchestrator)
        obj.oob_dist_utils = types.SimpleNamespace(all_gather_object=mock.Mock(side_effect=lambda value: [value, None]))
        block = types.SimpleNamespace(name='a')
        activities = [{'stream_id': 1, 'start_ns': 0}, {'stream_id': 1, 'start_ns': 1}]
        with self.assertRaisesRegex(RuntimeError, 'activity attribution failed'):
            obj._measure_blocks_performance(activities, [block], {'a': {1: 1}})
        error = obj.oob_dist_utils.all_gather_object.call_args.args[0]
        self.assertEqual(error['activity_counts'][1], {'expected': 1, 'captured': 2})

    def test_barrier_activities_do_not_affect_workload_latency(self):
        obj = object.__new__(self.orchestrator.Orchestrator)
        obj.rank = 0
        obj.oob_dist_utils = types.SimpleNamespace(all_gather_object=lambda value: [value])
        axis = types.SimpleNamespace(name='x', team_size=lambda: 2)
        block = self.make_comm_cls().from_api_params(axis, 'a', {'vector_size': 8}, Tensor(16))
        activities = [
            {'stream_id': 1, 'start_ns': 100, 'end_ns': 110},
            {'stream_id': 9, 'start_ns': 0, 'end_ns': 10000},
        ]
        results = obj._measure_blocks_performance(activities, [block], {'a': {1: 1}})
        self.assertEqual(results.iloc[0].avg_latency_ns, 10)

    def overlap_report(self, rank_intervals, axis_names=('x', 'y')):
        obj = object.__new__(self.orchestrator.Orchestrator)
        obj.rank = 0
        gemm = type('OverlapGemm', (self.block.GEMMBlock,), {'run': lambda self: None})
        blocks = [gemm.from_api_params(types.SimpleNamespace(name=axis), name,
                                      {'mat_a_shape': [2, 2], 'mat_b_shape': [2, 2]}, Tensor(8))
                  for axis, name in zip(axis_names, ('a', 'b'))]
        rank_perfs = []
        activities = []
        for rank, iterations in enumerate(rank_intervals):
            perfs = []
            for iteration, intervals in enumerate(iterations):
                for stream, (block, (start, end)) in enumerate(zip(blocks, intervals), 1):
                    perfs.append({'block_name': block.name, 'block_size': block.get_block_size_str(),
                                  'axis_name': block.axis.name, 'duration_ns': end - start,
                                  'start_ns': start, 'end_ns': end, 'rank': rank,
                                  'bandwidth_gbps': None, 'iter': iteration})
                    if rank == 0:
                        activities.append({'stream_id': stream, 'start_ns': start, 'end_ns': end})
            rank_perfs.append(perfs)
        obj.oob_dist_utils = types.SimpleNamespace(
            all_gather_object=lambda value: [None] * len(rank_intervals) if value is None else rank_perfs)
        return obj._measure_blocks_performance(activities, blocks, {'a': {1: 1}, 'b': {2: 1}},
                                              iters=len(rank_intervals[0]))

    def test_serialized_ranks_do_not_create_artificial_overlap(self):
        results = self.overlap_report([
            [[(0, 10), (10, 20)]],
            [[(10, 20), (0, 10)]],
        ])
        self.assertEqual(results.avg_overlap_pct.tolist(), [0, 0])

    def test_mean_overlap_is_independent_of_clock_offset_and_rank_duration(self):
        for offset in (1000, 10**15):
            with self.subTest(offset=offset):
                results = self.overlap_report([
                    [[(0, 10), (0, 10)]],
                    [[(offset, offset + 100), (offset + 100, offset + 200)]],
                ])
                self.assertEqual(results.avg_overlap_pct.tolist(), [50, 50])

    def test_rank_overlap_is_averaged_across_iterations(self):
        results = self.overlap_report([
            [[(0, 10), (0, 10)], [(100, 110), (110, 120)]],
            [[(1000, 1010), (1000, 1010)], [(1100, 1110), (1110, 1120)]],
        ])
        self.assertEqual(results.avg_overlap_pct.tolist(), [50, 50])

    def test_partial_and_zero_span_overlap(self):
        results = self.overlap_report([
            [[(0, 10), (5, 15)]],
            [[(1000, 1000), (1000, 1000)]],
        ])
        for percentage in results.avg_overlap_pct:
            self.assertAlmostEqual(percentage, 100 / 6)

    def test_multiple_blocks_on_one_axis_do_not_report_overlap(self):
        results = self.overlap_report([[[ (0, 10), (0, 10) ]]], axis_names=('x', 'x'))
        self.assertTrue(results.avg_overlap_pct.isna().all())


if __name__ == '__main__':
    unittest.main()
