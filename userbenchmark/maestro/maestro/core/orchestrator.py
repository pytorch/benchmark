"""The orchestrator is the main class and orchestrates the execution of a pattern

It is responsible for creating blocks based on the defined pattern, run them and measure their performance.
"""
from typing import Union, Callable, Optional
from collections import defaultdict
import statistics as st
import torch
import pandas as pd

from core.block import BlockRegistry, CommBlock, GEMMBlock
from core.axis import Axis
from core.profilers import profiler_ctx
from core.utils.distributed import get_rank_and_world_size, _DistUtils, root_print, dist_print, _TorchNCCLDistUtils, is_root
from core.utils.logging import get_logger, get_root_rank_logger



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
        """
        self.patterns = patterns
        self.warmup_iters = warmup_iters
        self.iters = iters
        self.axis_cls = axis_cls
        self.profiler = profiler
        self.latency_precision = latency_precision
        self.stop_on_error = stop_on_error
        self.register_buffers = register_buffers

        # Create axes as instances of axis_cls based on axis_cfg
        self.axes = {
            axis_name: self.axis_cls(groups, name=axis_name) for axis_name, groups in axes_cfg.items()
        }

        self._tensors_pool = {}

        self.oob_dist_utils = oob_dist_utils or self.axis_cls.dist_utils
        self.rank, self.world_size = get_rank_and_world_size()

        if self.world_size == 1:
            raise ValueError("Maestro is designed to run on multiple ranks, please use torchrun or slurm to launch the program.")

        self._prepare_tensors_pool()

    def run_all_patterns(self):
        """Run all the patterns"""
        all_results_df = pd.DataFrame() if is_root() else None
        for pattern in self.patterns:
            try:
                results_df = self.run_pattern(pattern)
            except Exception as e:
                if self.stop_on_error:
                    raise
                logger.exception(f"Error running pattern {pattern['name']}, skipping it, if you want to stop on error, set stop_on_error to True in YAML. Error: {e}")
            else:
                if is_root():
                    results_df["pattern"] = pattern["name"]
                    all_results_df = pd.concat([all_results_df, results_df])
        root_print("\n")
        if is_root():
            return all_results_df
        return

    def run_pattern(self, pattern: dict) -> Union[None, pd.DataFrame]:
        """Return results if rank=root, otherwise None"""
        if not pattern.get("ops"):
            raise ValueError("Pattern must contain at least one operation")

        rlogger.debug(f"Running pattern {pattern['name']}, creating blocks.")
        blocks = self._create_blocks(pattern)

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

        if not is_root():
            return

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
                f"{row.avg_latency_ns / 1e6:.1f}",
                f"{row.min_latency_ns / 1e6:.1f}",
                f"{row.max_latency_ns / 1e6:.1f}",
                f"{row.p99_latency_ns / 1e6:.1f}",
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
        """Measure the performance of each block, averaged across the ranks
        Params:
            results: The results of the profiler
            blocks: The blocks to measure the performance of
            streams_and_activities_per_block: A dictionary in the format {block_name: {stream_id: num_activities_on_this_stream}}
            iters: Number of iterations recorded, they should be the same

        Returns a df with performance for each block, where the duration has been averaged across the ranks then across the iterations
        duration by iteration is the max duration across the ranks
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

        # Overlap percentage calculation - only if there is one block per axis
        # Measures the percentage of total execution time where ALL blocks were running together
        df["overlap_pct"] = None

        if one_block_per_axis:
            for iter_ix, iter_df in df.groupby("iter"):
                # Find the overlap window: time when ALL blocks are running
                overlap_start = iter_df["start_ns"].max()  # latest start
                overlap_end = (iter_df["start_ns"] + iter_df["duration_ns"]).min()  # earliest end

                if overlap_end > overlap_start:
                    overlap_time = overlap_end - overlap_start
                else:
                    overlap_time = 0

                # Total time from first start to last end
                total_start = iter_df["start_ns"].min()
                total_end = (iter_df["start_ns"] + iter_df["duration_ns"]).max()
                total_time = total_end - total_start

                if total_time > 0:
                    overlap_pct = (overlap_time / total_time) * 100
                else:
                    overlap_pct = 0

                df.loc[df["iter"] == iter_ix, "overlap_pct"] = overlap_pct

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
        """Execute all the blocks <iters> times"""
        start_event = self.axis_cls.create_event()

        # Make all axes wait for the start event
        for axis in self.axes.values():
            axis.wait_event(start_event)

        self.oob_dist_utils.barrier()

        barrier_handle = None

        # Enqueue blocks
        axes = set(block.axis for block in blocks)
        for _ in range(iters):
            # Run blocks
            for block in blocks:
                block.run()

            # Make main stream wait for blocks completion
            for axis in axes:
                torch.cuda.current_stream().wait_stream(axis.stream)

            # Barrier
            barrier_handle = self.oob_dist_utils.barrier(async_op=True)
            assert barrier_handle is not None, "Barrier handle should not be None"

            # Make everyone wait for the barrier
            for axis in axes:
                with axis.use_axis():
                    barrier_handle.block_current_stream()

        # Start execution
        self.axis_cls.record_event(start_event)

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
            for op in pattern["ops"]:
                BlockCls = BlockRegistry.get_block_cls(op["block"])
                tensor_size = BlockCls.get_tensor_size(op)
                if tensor_size > 0:
                    tensor_sizes.append(tensor_size)
        if not tensor_sizes:
            raise ValueError("No positive tensor size found across patterns operations, please check your pattern configuration")

        if self.register_buffers:
            if not isinstance(self.axis_cls.dist_utils, _TorchNCCLDistUtils):
                raise ValueError(f"Register buffers is only supported with Torch-NCCL distributed utils, not {self.axis_cls.dist_utils.__class__.__name__}")
            # Force NCCL communicator initialization before mem pool creation;
            # even with device_id in init_process_group, the communicator may
            # still be lazily created and MemPool requires it up-front.
            self.oob_dist_utils.barrier()
            backend = self.axis_cls.dist_utils.get_world_group()._get_backend(torch.device('cuda'))
            tensors_mem_pool = torch.cuda.MemPool(backend.mem_allocator)
            with torch.cuda.use_mem_pool(tensors_mem_pool):
                t = torch.ones(max(tensor_sizes), device="cuda", dtype=dtype)
            backend.register_mem_pool(tensors_mem_pool)
        else:
            t = torch.ones(max(tensor_sizes), device="cuda", dtype=dtype)

            # For now dtype is hardcoded
        self._tensors_pool["main"] = t


    def _create_blocks(self, pattern: dict):
        """Create the blocks for the pattern"""
        blocks = []
        _blocks_names = set()
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
            block = BlockCls.from_api_params(axis=axis, name=block_name, api_params=op, cached_tensor=self._tensors_pool["main"])
            axis_name = op["axis"]
            blocks.append(block)

        return blocks

    def destroy(self):
        """Destroy the orchestrator"""
        self.oob_dist_utils.destroy()
        self.axis_cls.destroy()

