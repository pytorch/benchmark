"""The orchestrator is the main class and orchestrates the execution of a pattern

It is responsible for creating blocks based on the defined pattern, run them and measure their performance.
"""
from typing import Union, Callable, Optional
from collections import defaultdict
import statistics as st
import torch
import pandas as pd

from .block import BlockRegistry, CommBlock, GEMMBlock
from .axis import Axis
from .profilers import profiler_ctx
from .utils.distributed import get_rank_and_world_size, _DistUtils, root_print, dist_print
from .utils.logging import get_logger, get_root_rank_logger



logger = get_logger(__name__)
rlogger = get_root_rank_logger()

class Orchestrator:
    def __init__(
        self,
        patterns: list[dict],
        profiler: "profiler_ctx",
        axis_cls: type(Axis),
        axes_cfg: dict[str, list[list[int]]],
        warmup_iters: int = 20,
        iters: int = 1,
        oob_dist_utils: _DistUtils = None,
        latency_precision: int = 3,
        stop_on_error: bool = False,
        register_buffers: bool = False,
        axis_envs: dict[str, dict[str, str]] | None = None,
        axis_params: dict[str, dict] | None = None,
    ):
        """
        patterns: Pattern to execute
        profiler: Profiler to use
        axis_cls: Axis class to use
        axes_cfg: Axes teams, a dictionary {axis_name: teams} where teams is a list of lists, each sublist i contains the ranks of the team i
        warmup_iters: Number of warmup iterations
        iters: Number of iterations
        oob_dist_utils: Out-of-band distributed utilities to use
            If not provided, the distributed utilities of the axis class will be used
        stop_on_error: Stop the execution if an error occurs in one of the patterns
        register_buffers: Register pytorch buffers (PyTorch only)
        axis_params: Process-group options per axis, such as traffic_class
        axis_envs: Experimental overrides during process-group creation only,
            enabled by MAESTRO_USE_PER_AXIS_ENV_VARS; may lead to undefined behavior
        """
        self.patterns = patterns
        self.warmup_iters = warmup_iters
        self.iters = iters
        self.axis_cls = axis_cls
        self.profiler = profiler
        self.latency_precision = latency_precision
        self.stop_on_error = stop_on_error
        self.register_buffers = register_buffers

        axis_envs = axis_envs or {}
        axis_params = axis_params or {}
        self.axes = {
            axis_name: self.axis_cls(groups, name=axis_name, envs=axis_envs.get(axis_name),
                                          params=axis_params.get(axis_name))
            for axis_name, groups in axes_cfg.items()
        }

        self._tensors_pool = {}
        self._torch_mem_pool = None
        self._registered_groups = []

        self.oob_dist_utils = oob_dist_utils or self.axis_cls.dist_utils
        self.rank, self.world_size = get_rank_and_world_size()

        if self.world_size == 1:
            raise ValueError("Maestro is designed to run on multiple ranks, please use torchrun or slurm to launch the program.")

        self._prepare_tensors_pool()
        from .utils.start_gate import CUDAStartGate
        self._start_gate = CUDAStartGate()

    def run_all_patterns(self):
        """Run all the patterns"""
        all_results_df = pd.DataFrame()
        for pattern in self.patterns:
            try:
                results_df = self.run_pattern(pattern)
            except Exception as e:
                if self.stop_on_error:
                    raise
                logger.exception(f"Error running pattern {pattern['name']}, skipping it, if you want to stop on error, set stop_on_error to True in YAML. Error: {e}")
            else:
                results_df["pattern"] = pattern["name"]
                all_results_df = pd.concat([all_results_df, results_df])
        root_print("\n")
        return all_results_df

    def run_pattern(self, pattern: dict) -> pd.DataFrame:
        """Return the aggregated performance results on every rank."""
        if not pattern.get("ops"):
            raise ValueError("Pattern must contain at least one operation")

        rlogger.debug(f"Running pattern {pattern['name']}, creating blocks.")
        blocks = self._create_blocks(pattern)

        # Initialize libraries and allocation caches before holding GPU streams.
        # A first-use allocation or library setup may synchronize the device,
        # which would deadlock if the host had not released the gate yet.
        for block in blocks:
            block.run()
            self.axis_cls.synchronize_all()

        # Warmup 20 times - this is independent of warmup_iters which are perf-related
        rlogger.debug(f"Default warmup")
        for _ in range(20):
            self._execute(blocks)

        # Identify which stream correspond to which block
        # as an axis can have multiple streams, a reason it can happen is that pytorch uses different streams
        # for NCCL collectives, internally managed, so even if you set a stream and run a collective, it will use a different stream.
        # Therefore we match each block to a stream
        # We can however consider that different axes won't overlap on the same stream
        rlogger.debug(f"Matching streams to blocks")
        streams_and_activities_per_block = self._match_streams_to_blocks(blocks)

        # Benchmark
        self.profiler.reset()
        rlogger.debug(f"Starting real benchmarking")

        with self.profiler(name=f"{pattern['name']}"):
            self._execute(blocks, iters=self.warmup_iters)

            self.profiler.reset()
            self._execute(blocks, iters=self.iters)

        results = self.profiler.get_results()
        rlogger.debug(f"Benchmarking done, reporting results")
        results_df = self._measure_blocks_performance(results, blocks, streams_and_activities_per_block, iters=self.iters)


        columns = ["Block", "Size (B)",  "Avg latency (ms)", "Min lat. (ms)", "Max lat. (ms)", "P99 lat. (ms)", "Avg BW (GB/s)"]
        col_space = 17
        root_print("\n")
        root_print(f"Pattern: {pattern['name']}")
        root_print(f"| ".join([col.ljust(col_space) for col in columns]))
        root_print("-" * (len(columns) * col_space + len(columns)))

        for row in results_df.itertuples():

            if pd.isna(row.avg_bandwidth_gbps):
                avg_bandwidth_gbps = "N/A"
            else:
                avg_bandwidth_gbps = f"{row.avg_bandwidth_gbps:.1f}"
            row = [
                row.block_name,
                f"{row.block_size}",
                f"{row.avg_latency_ns / 1e6:.{self.latency_precision}f}",
                f"{row.min_latency_ns / 1e6:.{self.latency_precision}f}",
                f"{row.max_latency_ns / 1e6:.{self.latency_precision}f}",
                f"{row.p99_latency_ns / 1e6:.{self.latency_precision}f}",
                avg_bandwidth_gbps,
                ]
            root_print(f"| ".join([str(cell).ljust(col_space) for cell in row]))

        if "avg_shared_bw_gbps" in results_df.columns:
            shared_series = results_df["avg_shared_bw_gbps"].dropna()
            if not shared_series.empty:
                shared_bw_gbps = shared_series.iloc[0]
                root_print(f"\nAvg shared BW (GB/s): {shared_bw_gbps:.1f}")

        if "avg_overlap_pct" in results_df.columns:
            overlap_series = results_df["avg_overlap_pct"].dropna()
            if not overlap_series.empty:
                overlap_pct = overlap_series.iloc[0]
                root_print(f"Avg overlap: {overlap_pct:.1f}%")

        return results_df

    def _measure_blocks_performance(self, results: list[dict], blocks: list, streams_and_activities_per_block: dict[str, dict[str, int]], iters: int = 1) -> pd.DataFrame:
        """Measure each block using the slowest rank per iteration
        Params:
            results: The results of the profiler
            blocks: The blocks to measure the performance of
            streams_and_activities_per_block: A dictionary in the format {block_name: {stream_id: num_activities_on_this_stream}}
            iters: Number of iterations recorded, they should be the same

        Returns per-block performance aggregated across iterations.
        Duration by iteration is the max duration across ranks.
        Overlap is calculated per rank, then averaged across ranks and iterations.
        bandwidth_gbps is calculated for each block based on the aggregated duration
        """
        # Group result by stream_id
        activity_per_stream = defaultdict(list)

        for result in results:
            stream_id = result.get("stream_id")
            if stream_id is None:
                continue
            activity_per_stream[stream_id].append(result)

        for activities in activity_per_stream.values():
            activities.sort(key=lambda x: x["start_ns"])

        expected_counts = defaultdict(int)
        unmatched = []
        for block in blocks:
            stream_counts = streams_and_activities_per_block.get(block.name, {})
            if not stream_counts:
                unmatched.append(block.name)
            for stream_id, count in stream_counts.items():
                expected_counts[stream_id] += count * iters
        # As in the original stream mapper, only streams identified with a
        # workload block contribute to measurements. Other streams may carry
        # barrier or allocation activity and are ignored.
        actual_counts = {stream_id: len(activity_per_stream[stream_id]) for stream_id in expected_counts}
        mismatches = {
            stream_id: {"expected": expected_counts[stream_id], "captured": actual_counts[stream_id]}
            for stream_id in expected_counts
            if expected_counts[stream_id] != actual_counts[stream_id]
        }
        local_error = {"unmatched_blocks": unmatched, "activity_counts": mismatches} if unmatched or mismatches else None
        # Capture has completed on every rank. Exchange validation status before
        # parsing so a rank with extra/missing activities cannot strand its peers.
        errors = self.oob_dist_utils.all_gather_object(local_error)
        if any(error is not None for error in errors):
            raise RuntimeError(f"Profiler activity attribution failed on ranks: {errors}")

        # It is ok to iterate over blocks as the blocks are sorted by start_time in every activities list
        my_perf_by_block = []

        for iter_ix in range(iters):
            for block in blocks:
                stream_id_to_num_activities = streams_and_activities_per_block.get(block.name, {})
                if not stream_id_to_num_activities:
                    raise RuntimeError(
                        f"Block {block.name} was not matched to any stream, this should not happen. "
                        "Please contact the developers with this message and the YAML configuration you used, details: "
                        f'{{"error_at": "_measure_blocks_performance", "block_ran": "{block.name}", "profiler": "{self.profiler.__class__.__name__}"}}'
                    )

                block_activities = []
                for stream_id, num_activities in stream_id_to_num_activities.items():
                    for activity_ix in range(num_activities):
                        try:
                            activity = activity_per_stream[stream_id].pop(0)
                        except IndexError:
                            raise RuntimeError(
                                f"Block {block.name} activity number {activity_ix+1}/{num_activities} on stream {stream_id} could not be captured,"
                                f"or has been confused with a prior block that couldnt be captured. "
                                f"The profiler didn't manage to capture all the activities of this block, it may be because one activity is "
                                f"of a type that is not supported by this profiler. Run with MAESTRO_LOG_LEVEL=DEBUG to see more info."
                            )

                        block_activities.append(activity)


                start_ns = min(a["start_ns"] for a in block_activities)
                end_ns = max(a["end_ns"] for a in block_activities)
                duration_ns = end_ns - start_ns

                perf = {
                    "block_name": block.name,
                    "block_size": block.get_block_size_str(),
                    "axis_name": block.axis.name,
                    "duration_ns": duration_ns,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "rank": self.rank,
                    "bandwidth_gbps": None,
                    "iter": iter_ix,
                    }

                # Performance metrics specific to the block type
                if isinstance(block, CommBlock):
                    # This bandwidth is specific to this block and not aggregated across the ranks, it will be recalculated later
                    bw = block.get_bus_bw(exec_time_sec=duration_ns / 1e9)
                    perf["bandwidth_gbps"] = bw

                my_perf_by_block.append(perf)

        # Collect other ranks performances
        perf_by_block_per_rank = self.oob_dist_utils.all_gather_object(my_perf_by_block)

        df_dat = []
        for rank, rank_perfs in enumerate(perf_by_block_per_rank):
            for block_perf in rank_perfs:
                dat = block_perf.copy()
                dat["rank"] = rank
                df_dat.append(dat)

        df = pd.DataFrame(df_dat)
        df.sort_values(by=["rank", "start_ns"], inplace=True)

        # Calculate overlap before aggregating timestamps across ranks. Each
        # interval uses start/end from one rank's own clock domain.
        overlap_by_iter = {}
        if len({block.axis.name for block in blocks}) == len(blocks):
            overlap_per_rank = defaultdict(list)
            for (_, iter_ix), rank_iter_df in df.groupby(["rank", "iter"]):
                overlap_time = max(0, rank_iter_df["end_ns"].min() - rank_iter_df["start_ns"].max())
                total_time = rank_iter_df["end_ns"].max() - rank_iter_df["start_ns"].min()
                overlap_pct = 100 * overlap_time / total_time if total_time > 0 else 0
                overlap_per_rank[iter_ix].append(overlap_pct)
            overlap_by_iter = {iter_ix: st.mean(percentages)
                               for iter_ix, percentages in overlap_per_rank.items()}

        # Aggregate across ranks
        # duration_ns is the max duration_ns
        # start_ns is the min start_ns
        # rank and end_ns disappear
        # bandwidth_gbps is recalculated for each block given the new duration_ns
        # block_name, block_size, axis are the same for all ranks
        df = df.groupby(["block_name", "iter"]).agg({
            "duration_ns": "max",
            "start_ns": "min",
            "block_size": "first",
            "axis_name": "first",
        }).reset_index()

        # Recalculate bandwidth_gbps for CommBlocks based on aggregated duration
        df["bandwidth_gbps"] = None
        for idx, row in df.iterrows():
            block = self.get_block_by_name(blocks, row["block_name"])
            if isinstance(block, CommBlock):
                df.at[idx, "bandwidth_gbps"] = block.get_bus_bw(exec_time_sec=row["duration_ns"] / 1e9)

        # Shared bandwidth calculation - only if:
        # - There is only one block per axis
        # - All the blocks are communication blocks
        df["shared_bw_gbps"] = None

        first_iter_blocks = [
            self.get_block_by_name(blocks, name)
            for name in df.loc[df["iter"] == 0, "block_name"]
        ]

        all_comm = all(isinstance(block, CommBlock) for block in first_iter_blocks)
        one_block_per_axis = df.loc[df["iter"] == 0, "axis_name"].is_unique

        if all_comm and one_block_per_axis:
            for iter_ix, iter_df in df.groupby("iter"):
                total_weighted_size = 0

                for _, row in iter_df.iterrows():
                    block = self.get_block_by_name(blocks, row["block_name"])

                    msg_size_gb = block.get_full_vector_size() / 1e9
                    team_size = block.axis.team_size()
                    factor = block.get_bus_bw_factor(team_size)

                    total_weighted_size += msg_size_gb * factor

                max_duration_sec = iter_df["duration_ns"].max() / 1e9
                shared_bw_gbps = total_weighted_size / max_duration_sec

                df.loc[df["iter"] == iter_ix, "shared_bw_gbps"] = shared_bw_gbps

        # Report only mean rank-local overlap, then average across iterations.
        df["overlap_pct"] = df["iter"].map(overlap_by_iter) if one_block_per_axis else None

        # Aggregate across iters
        df = df.groupby(["block_name"]).agg(
            avg_latency_ns=("duration_ns", "mean"),
            min_latency_ns=("duration_ns", "min"),
            max_latency_ns=("duration_ns", "max"),
            p99_latency_ns=("duration_ns", lambda s: s.quantile(0.99)),
            avg_bandwidth_gbps=("bandwidth_gbps", "mean"),
            avg_shared_bw_gbps=("shared_bw_gbps", "mean"),
            avg_overlap_pct=("overlap_pct", "mean"),
            start_ns=("start_ns", "first"),
            block_size=("block_size", "first"),
            axis_name=("axis_name", "first"),
        ).reset_index()

        df.sort_values(by="start_ns", inplace=True)
        # Remove start_ns now that the rows are sorted
        df.drop(columns=["start_ns"], inplace=True)

        return df

    def get_block_by_name(self, blocks: list, name: str):
        """Return the block by its name"""
        for block in blocks:
            if block.name == name:
                return block
        return None

    def _match_streams_to_blocks(self, blocks: list) -> dict[str, dict[str, int]]:
        """Match the streams to the blocks by executing them and checking the stream_id of the captured activity
        Returns: A dictionary in the format {block_name: {stream_id: num_activities_on_this_stream}}
        This dictionary covers all the blocks, if one block cannot be matched to a stream, an error will be raised.
        """
        streams_and_activities_per_block = {}
        for block in blocks:
            self.profiler.reset()
            with self.profiler(nosave=True):
                block.run()
                self.axis_cls.synchronize_all()

            results = self.profiler.get_results()

            if not results:
                raise RuntimeError(
                    f"Could not match stream to block {block.name} - the activity was not captured successfully by the profiler. "
                    f"This could be due to the fact that this operation doesn't launch a type of CUDA operation caught by this profiler, "
                    f"for example collectives with only one rank sometimes don't launch any operation at all as it would be useless."
                    f"Please contact the developers with this message and the YAML configuration you used, details: "
                    f'{{"error_at": "_match_streams_to_blocks", "block_ran": "{block.name}", "profiler": "{self.profiler.__class__.__name__}"}}'
                )

            streams_and_activities_per_block[block.name] = {}

            for result in results:
                stream_id = result["stream_id"]
                # Increment stream_id
                streams_and_activities_per_block[block.name][stream_id] = streams_and_activities_per_block[block.name].get(stream_id, 0) + 1

        return streams_and_activities_per_block

    def _execute(self, blocks: list, iters: int = 1):
        """Enqueue all axes behind a recorded, host-released start event."""
        if iters == 0:
            return
        self.oob_dist_utils.barrier()
        axes = set(block.axis for block in blocks)

        with self._start_gate.hold(axes):
            for _ in range(iters):
                for block in blocks:
                    block.run()

                # Fence each iteration on every participating axis. The host
                # release is independent of this stream, avoiding a gate cycle.
                for axis in axes:
                    torch.cuda.current_stream().wait_stream(axis.stream)

                barrier_handle = self.oob_dist_utils.barrier(async_op=True)
                assert barrier_handle is not None, "Barrier handle should not be None"
                for axis in axes:
                    with axis.use_axis():
                        barrier_handle.block_current_stream()

        self.axis_cls.synchronize_all()

    def _prepare_tensors_pool(self):
        """Initialize self._tensors_pool
        The tensor pool is a pool of tensors that can be re-used so that we don't have to allocate and deallocate tensors for each block.
        For now only one pool exist: 'main' but it's possible to allocate more, for example for those use cases:
           - Check data correctness (that dst_buf is filled with the correct data)
           - Use different dtypes for different blocks (for example, float16 for GEMM and int8 for all_gather)
        """
        dtype = torch.float16

        tensor_sizes = []
        for pattern in self.patterns:
            pattern_size = 0
            for op in pattern["ops"]:
                BlockCls = BlockRegistry.get_block_cls(op["block"])
                tensor_size = self._get_op_tensor_size(BlockCls, op)
                if tensor_size > 0:
                    pattern_size += tensor_size
            tensor_sizes.append(pattern_size)
        if not tensor_sizes or max(tensor_sizes) <= 0:
            raise ValueError("No positive tensor size found across patterns operations, please check your pattern configuration")

        if self.register_buffers:
            t = self._prepare_nccl_registered_tensor(max(tensor_sizes), dtype)
        else:
            t = torch.ones(max(tensor_sizes), device="cuda", dtype=dtype)

            # For now dtype is hardcoded
        self._tensors_pool["main"] = t

    def _prepare_nccl_registered_tensor(self, tensor_size: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Allocate Maestro's tensor pool from PyTorch's NCCL allocator and
        register it on the axis process groups that run communication blocks.
        """
        from .axis import TorchAxis
        groups = [
            axis.get_process_group()
            for axis in self.axes.values()
            if isinstance(axis, TorchAxis)
        ]
        if not groups:
            raise RuntimeError("register_buffers=True but no Torch axis was found.")

        device = torch.device("cuda", torch.cuda.current_device())
        if self._torch_mem_pool is None:
            backend = groups[0]._get_backend(device)
            self._torch_mem_pool = torch.cuda.MemPool(backend.mem_allocator)

        with torch.cuda.use_mem_pool(self._torch_mem_pool):
            tensor = torch.ones(tensor_size, device="cuda", dtype=dtype)
        torch.cuda.synchronize()

        for group in groups:
            group._get_backend(device).register_mem_pool(self._torch_mem_pool)

        self._registered_groups.extend(groups)
        rlogger.info(f"Registered Maestro NCCL tensor pool for {len(groups)} axis process group(s)")
        return tensor

    def _get_op_tensor_size(self, block_cls, op: dict) -> int:
        if issubclass(block_cls, CommBlock):
            return block_cls.get_tensor_size(op, self.axes[op["axis"]])
        return block_cls.get_tensor_size(op)

    def _create_blocks(self, pattern: dict):
        """Create the blocks for the pattern"""
        blocks = []
        _blocks_names = set()
        tensor_offset = 0
        for op in pattern["ops"]:
            BlockCls = BlockRegistry.get_block_cls(op["block"])

            # Get axis
            axis_name = op["axis"]
            axis = self.axes.get(axis_name)
            if axis is None:
                raise ValueError(f"Axis {axis_name} was not defined in the pattern")

            # Get block name and check it doesn't already exist
            block_name = op.get("name", f"block_{len(blocks)}_{BlockCls.registry_name}")
            if block_name in _blocks_names:
                raise ValueError(f"Two blocks found with the name {block_name}, use a different name for every block in the same pattern")
            _blocks_names.add(block_name)

            # Instanciate block
            tensor_size = max(0, self._get_op_tensor_size(BlockCls, op))
            cached_tensor = self._tensors_pool["main"][tensor_offset:tensor_offset + tensor_size]
            tensor_offset += tensor_size
            # Communication starts at zero so repeated in-place reductions remain
            # finite. Initialize before warmup, outside the measured region.
            if issubclass(BlockCls, CommBlock):
                cached_tensor.zero_()
            else:
                cached_tensor.fill_(1)
            block = BlockCls.from_api_params(axis=axis, name=block_name, api_params=op, cached_tensor=cached_tensor)
            axis_name = op["axis"]
            blocks.append(block)

        # Initialization uses the current stream; complete it before axes launch
        # work on their own streams. This happens outside profiling.
        self.axis_cls.synchronize_all()
        return blocks

    def destroy(self):
        """Destroy the orchestrator"""
        self._start_gate.close()
        self._cleanup_registered_pool()
        self.oob_dist_utils.destroy()
        self.axis_cls.destroy()

    def _cleanup_registered_pool(self):
        """Deregister the tensor pool before process groups are destroyed."""
        torch_mem_pool = self._torch_mem_pool
        if torch_mem_pool is None:
            return

        self.axis_cls.synchronize_all()
        device = torch.device("cuda", torch.cuda.current_device())
        for group in self._registered_groups:
            try:
                group._get_backend(device).deregister_mem_pool(torch_mem_pool)
            except Exception as exc:
                logger.warning(f"Failed to deregister NCCL mem pool: {exc}")
        self.axis_cls.synchronize_all()

        self._tensors_pool.clear()
        self._torch_mem_pool = None
        self._registered_groups = []
