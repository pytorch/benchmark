#!/usr/bin/env python3
"""Generate model_details.json for modelrunner from .pt2 ExportedPrograms.

Reads every <name>.pt2 in exported_pt2/, extracts input shapes and dtypes
from ep.example_inputs and output count from a forward pass, writes a
single model_details.json that modelrunner consumes via --input-dir.

Usage:
  ./gen-model-details.py                  # scan all .pt2 in this directory
  ./gen-model-details.py --only resnet50  # update a single entry
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUT_FILE = HERE / "model_details.json"

DTYPE_MAP = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float64: "float64",
    torch.int8:    "int8",
    torch.int16:   "int16",
    torch.int32:   "int32",
    torch.int64:   "int64",
    torch.uint8:   "uint8",
    torch.bool:    "bool",
}


def details_for(pt2: Path) -> dict | None:
    try:
        ep = torch.export.load(str(pt2))
    except Exception as e:
        print(f"  FAIL load: {pt2.name}: {type(e).__name__}: {e}")
        return None

    args, kwargs = ep.example_inputs
    if kwargs:
        print(f"  WARN {pt2.name}: has kwargs ({sorted(kwargs)}); ignoring")

    tensors = [t for t in args if isinstance(t, torch.Tensor)]
    if len(tensors) != len(args):
        print(f"  WARN {pt2.name}: non-tensor args in example_inputs; using only tensor positions")

    entry: dict = {"num_inputs": len(tensors)}
    for i, t in enumerate(tensors):
        entry[f"shape{i}"] = list(t.shape)
        entry[f"dtype{i}"] = DTYPE_MAP.get(t.dtype, str(t.dtype).replace("torch.", ""))

    # Count outputs by running the module once with example inputs.
    try:
        with torch.no_grad():
            out = ep.module()(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            entry["num_outputs"] = 1
        elif isinstance(out, (tuple, list)):
            entry["num_outputs"] = len(out)
        elif isinstance(out, dict):
            entry["num_outputs"] = len(out)
        else:
            entry["num_outputs"] = 1
    except Exception as e:
        print(f"  WARN {pt2.name}: forward failed ({type(e).__name__}); defaulting num_outputs=1")
        entry["num_outputs"] = 1

    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=[], help="Model name (repeatable). Updates only these entries.")
    args = ap.parse_args()

    # Start from existing file if present so partial updates don't drop entries.
    if OUT_FILE.exists():
        with OUT_FILE.open() as f:
            details = json.load(f)
    else:
        details = {}

    pt2_files = sorted(HERE.glob("*.pt2"))
    if args.only:
        wanted = set(args.only)
        pt2_files = [p for p in pt2_files if p.stem in wanted]
        missing = wanted - {p.stem for p in pt2_files}
        if missing:
            print(f"WARN: --only names with no .pt2: {sorted(missing)}", file=sys.stderr)

    print(f"Scanning {len(pt2_files)} .pt2 file(s)…")
    for pt2 in pt2_files:
        print(f"  {pt2.name}")
        entry = details_for(pt2)
        if entry is not None:
            details[pt2.stem] = entry

    with OUT_FILE.open("w") as f:
        json.dump(details, f, indent=4, sort_keys=True)
    print(f"\nWrote {OUT_FILE} ({len(details)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
