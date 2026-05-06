"""
Whisper Medium Benchmark
========================
Benchmarks OpenAI Whisper medium encoder inference using PyTorch.

Measures:
  - Average latency per inference (CPU fp32, CUDA fp16)
  - Encoder FLOPs (via torch.utils.flop_counter.FlopCounterMode)
  - CPU-to-GPU speedup ratio

Input shape follows the existing hf_Whisper torchbenchmark model:
  (batch=1, mel_bins=80, time_frames=3000) — 30 seconds of audio at 100 fps.

Usage
-----
Via userbenchmark router:
    python run_benchmark.py whisper [--no-cuda] [--warmup N] [--iters N]

Direct execution:
    python userbenchmark/whisper/run.py [--no-cuda] [--warmup N] [--iters N]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Constants — aligned with torchbenchmark/models/hf_Whisper conventions
# ---------------------------------------------------------------------------
MODEL_NAME = "openai/whisper-medium"

# Mel-spectrogram shape: (batch, mel_bins, time_frames)
# Matches hf_Whisper's example_inputs construction: (batch_size, 80, 3000)
INPUT_SHAPE = (1, 80, 3000)

# Benchmark iterations — reduced vs. default WARMUP_ROUNDS=10/BENCHMARK_ITERS=15
# because Whisper medium encoder is large (~300M params) and slow on CPU.
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 10

NS_PER_MS = 1_000_000.0

BM_NAME = "whisper"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: str) -> "WhisperForConditionalGeneration":
    """Load Whisper medium from HuggingFace Hub and move to device.

    Mirrors hf_Whisper's approach:
      - CPU  → fp32
      - CUDA → fp16  (DEFAULT_EVAL_CUDA_PRECISION = "fp16" in hf_Whisper)
    """
    from transformers import WhisperForConditionalGeneration

    print(f"  Loading {MODEL_NAME} ...", end=" ", flush=True)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    except Exception as exc:
        print(f"\nFailed to load model: {exc}", file=sys.stderr)
        print(
            "Ensure you have internet access or a cached copy of the model.\n"
            "Install dependencies: pip install transformers>=4.23.0",
            file=sys.stderr,
        )
        sys.exit(1)

    model.eval()
    model = model.to(device)

    # Match hf_Whisper: use fp16 on CUDA to reflect real-world inference.
    if device == "cuda":
        model = model.half()

    print("done.")
    return model


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def measure_latency(
    model: "WhisperForConditionalGeneration",
    input_features: torch.Tensor,
    device: str,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
) -> List[float]:
    """Return per-inference latencies in milliseconds.

    Follows torchbenchmark/util/experiment/metrics.py::get_latencies():
      - warmup loop (discarded)
      - timed loop with synchronize() bracketing every iteration
      - returns list of float latencies in ms
    """
    encoder = model.model.encoder

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _sync(device)
            encoder(input_features)
            _sync(device)

    # Timed benchmark
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(iters):
            _sync(device)
            t0 = time.time_ns()
            encoder(input_features)
            _sync(device)
            t1 = time.time_ns()
            latencies.append((t1 - t0) / NS_PER_MS)

    return latencies


# ---------------------------------------------------------------------------
# FLOPS measurement
# ---------------------------------------------------------------------------

def measure_flops() -> Optional[int]:
    """Count encoder FLOPs for one forward pass using FlopCounterMode.

    Runs on CPU in fp32 — FlopCounterMode is device/dtype-agnostic but CPU
    makes the import path reliable across all PyTorch builds.

    Returns None if FlopCounterMode is unavailable (PyTorch < 2.0).
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        print(
            "  Warning: torch.utils.flop_counter not available (PyTorch >= 2.0 required).",
            file=sys.stderr,
        )
        return None

    # FLOPs measurement always uses a fresh CPU fp32 copy so results are
    # independent of the device/dtype used for latency benchmarking.
    from transformers import WhisperForConditionalGeneration

    print(f"  Loading CPU fp32 copy for FLOPs measurement ...", end=" ", flush=True)
    try:
        flop_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    except Exception as exc:
        print(f"\n  FLOPs skipped — could not load model: {exc}", file=sys.stderr)
        return None
    flop_model.eval()
    print("done.")

    cpu_input = torch.randn(INPUT_SHAPE, dtype=torch.float32)

    flop_counter = FlopCounterMode(display=False)
    with torch.no_grad(), flop_counter:
        flop_model.model.encoder(cpu_input)

    total_flops = sum(flop_counter.flop_counts["Global"].values())
    del flop_model
    return int(total_flops)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_ms(ms: float) -> str:
    return f"{ms:>10.2f} ms"


def _fmt_flops(flops: int) -> str:
    gflops = flops / 1e9
    return f"{flops:,}  (~{gflops:.1f} GFLOPs)"


def print_results(
    cpu_latencies: List[float],
    cuda_latencies: Optional[List[float]],
    flops: Optional[int],
) -> None:
    """Print a human-readable benchmark summary table."""
    sep = "-" * 72

    print()
    print("=" * 72)
    print(f"  Whisper Medium Benchmark  ({MODEL_NAME})")
    print(f"  Input : {INPUT_SHAPE}  (batch, mel_bins, time_frames)")
    print(f"  Task  : encoder-only inference (model.model.encoder)")
    print("=" * 72)

    # Header
    print(f"\n{'Device':<8}  {'Dtype':<6}  {'Avg':>10}  {'Std':>9}  {'Min':>10}  {'Max':>10}")
    print(sep)

    # CPU row
    cpu_mean = statistics.mean(cpu_latencies)
    cpu_std = statistics.stdev(cpu_latencies) if len(cpu_latencies) > 1 else 0.0
    cpu_min = min(cpu_latencies)
    cpu_max = max(cpu_latencies)
    print(
        f"{'CPU':<8}  {'fp32':<6}  {cpu_mean:>10.2f}  {cpu_std:>8.2f}  "
        f"{cpu_min:>10.2f}  {cpu_max:>10.2f}  ms"
    )

    # CUDA row (optional)
    if cuda_latencies:
        cuda_mean = statistics.mean(cuda_latencies)
        cuda_std = statistics.stdev(cuda_latencies) if len(cuda_latencies) > 1 else 0.0
        cuda_min = min(cuda_latencies)
        cuda_max = max(cuda_latencies)
        print(
            f"{'CUDA':<8}  {'fp16':<6}  {cuda_mean:>10.2f}  {cuda_std:>8.2f}  "
            f"{cuda_min:>10.2f}  {cuda_max:>10.2f}  ms"
        )

    print(sep)

    # FLOPs
    if flops is not None:
        print(f"\n  Encoder FLOPs : {_fmt_flops(flops)}")
    else:
        print("\n  Encoder FLOPs : N/A")

    # Speedup
    if cuda_latencies:
        speedup = statistics.mean(cpu_latencies) / statistics.mean(cuda_latencies)
        print(f"  CPU→GPU Speedup : {speedup:.1f}x")
    else:
        print("  CPU→GPU Speedup : N/A  (CUDA not available / skipped)")

    print()


# ---------------------------------------------------------------------------
# Metrics dict for JSON output
# ---------------------------------------------------------------------------

def build_metrics(
    cpu_latencies: List[float],
    cuda_latencies: Optional[List[float]],
    flops: Optional[int],
) -> Dict[str, float]:
    """Assemble the flat metrics dict written to the userbenchmark JSON file."""
    metrics: Dict[str, float] = {
        "cpu_latency_mean_ms": round(statistics.mean(cpu_latencies), 4),
        "cpu_latency_std_ms": round(
            statistics.stdev(cpu_latencies) if len(cpu_latencies) > 1 else 0.0, 4
        ),
        "cpu_latency_min_ms": round(min(cpu_latencies), 4),
        "cpu_latency_max_ms": round(max(cpu_latencies), 4),
    }

    if flops is not None:
        metrics["encoder_flops"] = float(flops)
        metrics["encoder_gflops"] = round(flops / 1e9, 3)

    if cuda_latencies:
        cuda_mean = statistics.mean(cuda_latencies)
        metrics["cuda_latency_mean_ms"] = round(cuda_mean, 4)
        metrics["cuda_latency_std_ms"] = round(
            statistics.stdev(cuda_latencies) if len(cuda_latencies) > 1 else 0.0, 4
        )
        metrics["cuda_latency_min_ms"] = round(min(cuda_latencies), 4)
        metrics["cuda_latency_max_ms"] = round(max(cuda_latencies), 4)
        metrics["cpu_to_cuda_speedup"] = round(
            statistics.mean(cpu_latencies) / cuda_mean, 2
        )

    return metrics


# ---------------------------------------------------------------------------
# Entry point — required by userbenchmark spec
# ---------------------------------------------------------------------------

def run(args: List[str] = []) -> None:
    """Run the Whisper medium benchmark.

    This is the required entry point for the userbenchmark framework.
    Invoked via: python run_benchmark.py whisper [args]
    """
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI Whisper medium encoder latency and FLOPs."
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Skip CUDA benchmark even if a GPU is available.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        metavar="N",
        help=f"Warmup iterations before timing (default: {DEFAULT_WARMUP}).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        metavar="N",
        help=f"Timed benchmark iterations (default: {DEFAULT_ITERS}).",
    )
    parser.add_argument(
        "--no-flops",
        action="store_true",
        help="Skip FLOPs measurement (saves time; FLOPs require a second model load).",
    )
    parsed = parser.parse_args(args)

    cuda_available = torch.cuda.is_available() and not parsed.no_cuda
    if parsed.no_cuda:
        print("CUDA benchmark skipped (--no-cuda).")
    elif not torch.cuda.is_available():
        print("CUDA not available — running CPU benchmark only.")

    # ------------------------------------------------------------------
    # CPU benchmark
    # ------------------------------------------------------------------
    print("\n[1/3] CPU benchmark (fp32)")
    cpu_model = load_model("cpu")
    cpu_input = torch.randn(INPUT_SHAPE, dtype=torch.float32)
    print(
        f"  Warmup {parsed.warmup} iters, then timing {parsed.iters} iters ...",
        flush=True,
    )
    cpu_latencies = measure_latency(
        cpu_model, cpu_input, "cpu", warmup=parsed.warmup, iters=parsed.iters
    )
    print(f"  Avg latency: {statistics.mean(cpu_latencies):.2f} ms")
    del cpu_model

    # ------------------------------------------------------------------
    # CUDA benchmark
    # ------------------------------------------------------------------
    cuda_latencies: Optional[List[float]] = None
    if cuda_available:
        print("\n[2/3] CUDA benchmark (fp16)")
        cuda_model = load_model("cuda")
        # fp16 input — matches model dtype on CUDA
        cuda_input = torch.randn(INPUT_SHAPE, dtype=torch.float16, device="cuda")
        print(
            f"  Warmup {parsed.warmup} iters, then timing {parsed.iters} iters ...",
            flush=True,
        )
        cuda_latencies = measure_latency(
            cuda_model, cuda_input, "cuda", warmup=parsed.warmup, iters=parsed.iters
        )
        print(f"  Avg latency: {statistics.mean(cuda_latencies):.2f} ms")
        del cuda_model
    else:
        print("\n[2/3] CUDA benchmark — skipped.")

    # ------------------------------------------------------------------
    # FLOPs measurement
    # ------------------------------------------------------------------
    flops: Optional[int] = None
    if not parsed.no_flops:
        print("\n[3/3] FLOPs measurement (CPU fp32, single forward pass)")
        flops = measure_flops()
        if flops is not None:
            print(f"  Encoder FLOPs: {_fmt_flops(flops)}")
    else:
        print("\n[3/3] FLOPs measurement — skipped (--no-flops).")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print_results(cpu_latencies, cuda_latencies, flops)

    # ------------------------------------------------------------------
    # JSON output (userbenchmark format)
    # ------------------------------------------------------------------
    # Import here so the script can also be run standalone without the
    # full torchbenchmark package on sys.path.
    try:
        _repo_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(_repo_root))
        from userbenchmark.utils import dump_output, get_output_json

        metrics = build_metrics(cpu_latencies, cuda_latencies, flops)
        output = get_output_json(BM_NAME, metrics)
        dump_output(BM_NAME, output)
        print(f"Metrics written to .userbenchmark/{BM_NAME}/metrics-<timestamp>.json")
    except Exception as exc:
        print(f"Warning: could not write JSON output — {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Allow direct execution: python userbenchmark/whisper/run.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run(sys.argv[1:])
