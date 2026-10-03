import os
import json
from core.utils.logging import get_logger, get_root_rank_logger
from pathlib import Path
from typing import Optional
from core.profilers.base import profiler_ctx

try:
    from cupti import cupti
    CUPTI_AVAILABLE = True
except ImportError:
    CUPTI_AVAILABLE = False


logger = get_logger(__name__)
rlogger = get_root_rank_logger()


class CuptiProfiler(profiler_ctx):
    """Wraps CUPTI profiling interface"""
    
    def __init__(self, output_dir=None):
        if not CUPTI_AVAILABLE:
            raise ImportError("CUPTI is not available. Please install cupti-python package.")

        super().__init__(output_dir)
        
        self.activities = [
            cupti.ActivityKind.CONCURRENT_KERNEL,
            cupti.ActivityKind.MEMCPY,
            cupti.ActivityKind.MEMSET,
            cupti.ActivityKind.MEMCPY2,
        ]

        cupti.activity_register_callbacks(self._func_buffer_requested, self._func_buffer_completed)

    
    def _func_buffer_requested(self):
        """See https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv432CUpti_BuffersCallbackRequestFunc"""
        buffer_size = 10 * 1024 * 1024      # 10MB
        max_num_records = 0                 # Means infinite number of records
        return buffer_size, max_num_records

    def _func_buffer_completed(self, activities: list):
        """See https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv433CUpti_BuffersCallbackCompleteFunc """
        results = []
        for activity in activities:
            try:
                if activity.kind == cupti.ActivityKind.KERNEL or activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                    results.append({
                        "kind": "kernel",
                        "name": activity.name,
                        "stream_id": activity.stream_id,
                        "start_ns": activity.start,
                        "end_ns": activity.end,
                        "completed_ns": activity.completed,
                        "duration_ns": activity.end - activity.start,
                    })
                elif activity.kind in [cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMCPY2]:
                    results.append({
                        "kind": "memcpy",
                        "name": "memcpy",
                        "stream_id": activity.stream_id,
                        "start_ns": activity.start,
                        "end_ns": activity.end,
                        "size": activity.bytes,
                        "src_kind": activity.src_kind,
                        "dst_kind": activity.dst_kind,
                        "duration_ns": activity.end - activity.start,
                    })
                elif activity.kind == cupti.ActivityKind.MEMSET:
                    results.append({
                        "kind": "memset",
                        "name": "memset",
                        "stream_id": activity.stream_id,
                        "start_ns": activity.start,
                        "end_ns": activity.end,
                        "size": activity.bytes,
                        "memory_kind": activity.memory_kind,
                        "duration_ns": activity.end - activity.start,
                    })
                else:
                    logger.info(f"Unsupported activity kind: {activity}")
            except Exception as e:
                logger.error(f"Error collecting CUPTI results: {e}")
                raise
        
        try:
            self._save_results(results)
        except Exception as e:
            logger.exception(f"Error saving CUPTI results: {e}")
            return

    def _enter(self):
        """Start CUPTI profiling"""
        for activity in self.activities:
            cupti.activity_enable(activity)
    
    def _exit(self, nosave: bool = False):
        """Stop CUPTI profiling and optionally save results
        
        Args:
            nosave: If True, skip saving trace
        """
        cupti.activity_flush_all(1)

        for activity in self.activities:
            cupti.activity_disable(activity)
    
    def _save_trace(self, output_file=None):
        """Save results in Chrome Trace Event Format (JSON)"""
        trace_events = []
        
        # Convert all collected results to Chrome Trace format
        for event in self.get_results():
            trace_event = {
                "name": event.get("name", "unknown"),
                "cat": event.get("kind", "unknown"),
                "ph": "X",  # Complete event (has duration)
                "ts": event["start_ns"] / 1000.0,  # Convert ns to microseconds
                "dur": event["duration_ns"] / 1000.0,  # Convert ns to microseconds
                "pid": self.rank,  # Use rank as process ID
                "tid": event["stream_id"],  # Use stream_id as thread ID
                "args": {
                    "completed_ns": event.get("completed_ns", 0),
                    "end_ns": event.get("end_ns", 0),
                }
            }
            trace_events.append(trace_event)
        
        trace_data = {
            "traceEvents": trace_events,
            "displayTimeUnit": "ns",
            "meta_user": f"rank_{self.rank}",
            "meta_world_size": self.world_size
        }
        
        with open(output_file, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        return output_file 