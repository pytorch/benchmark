#!/usr/bin/env python3
"""Export pytorch/benchmark models to .tflite via litert_torch.

Data-driven from ../exported_pt2/torchbench-models.yaml. Only attempts
models that have a PASS entry in the pt2 export-status.csv (so we don't
waste time on models that can't even be exported to pt2).

Usage:
  ./export-torchbench-litert-batch.py                    # export all pt2-passing models
  ./export-torchbench-litert-batch.py --only resnet50    # one model
  ./export-torchbench-litert-batch.py --families vision,timm
  ./export-torchbench-litert-batch.py --retry-failed     # re-run FAIL rows
  ./export-torchbench-litert-batch.py --force            # ignore prior status
  ./export-torchbench-litert-batch.py --list-only
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import litert_torch
from importlib import util as _util

REPO_ROOT = Path(__file__).resolve().parent.parent
PT2_DIR = REPO_ROOT / "exported_pt2"
HERE = REPO_ROOT / "exported_litert"
DEFAULT_MANIFEST = PT2_DIR / "torchbench-models.yaml"
STATUS_CSV = HERE / "export-status.csv"
PT2_STATUS_CSV = PT2_DIR / "export-status.csv"

# Import _apply_model_patches from the sibling pt2 exporter
_spec = _util.spec_from_file_location("_pt2_exporter", PT2_DIR / "export-torchbench-batch.py")
_pt2 = _util.module_from_spec(_spec)
_spec.loader.exec_module(_pt2)
_apply_model_patches = _pt2._apply_model_patches


class TensorOutputWrapper(nn.Module):
    """Flatten HF dataclass outputs to plain tensors; convert dict inputs to positional."""

    def __init__(self, model: nn.Module, kwarg_names: list[str] | None = None) -> None:
        super().__init__()
        self.model = model
        self.kwarg_names = kwarg_names

    def forward(self, *args, **kwargs):
        if self.kwarg_names is not None and not kwargs:
            kwargs = dict(zip(self.kwarg_names, args))
            args = ()
        out = self.model(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "to_tuple"):
            flat = tuple(t for t in out.to_tuple() if isinstance(t, torch.Tensor))
            return flat[0] if len(flat) == 1 else flat
        if isinstance(out, dict):
            flat = tuple(v for v in out.values() if isinstance(v, torch.Tensor))
            return flat[0] if len(flat) == 1 else flat
        if isinstance(out, (tuple, list)):
            flat = tuple(t for t in out if isinstance(t, torch.Tensor))
            return flat[0] if len(flat) == 1 else flat
        return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--families", help="Comma-separated family filter")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list-only", action="store_true")
    return ap.parse_args()


def load_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        return yaml.safe_load(f)["models"]


def load_csv_status(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {row["name"]: row["status"] for row in csv.DictReader(f)}


def write_status(path: Path, rows: list[dict]) -> None:
    fields = ["name", "family", "status", "elapsed_s", "out_path", "error"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def filter_models(models: list[dict], args: argparse.Namespace,
                  prior: dict[str, str], pt2_pass: set[str]) -> list[dict]:
    out = [m for m in models if not m.get("skip_reason") and m["name"] in pt2_pass]
    if args.only:
        wanted = set(args.only)
        out = [m for m in out if m["name"] in wanted]
    if args.families:
        fams = set(args.families.split(","))
        out = [m for m in out if m["family"] in fams]
    if args.retry_failed:
        out = [m for m in out if prior.get(m["name"]) in ("FAIL", "TIMEOUT", None)]
    elif not args.force and not args.only:
        out = [m for m in out if not (
            prior.get(m["name"]) == "PASS" and
            (args.out_dir / f"{m['name']}.tflite").exists()
        )]
    return out


def export_one(model: dict, out_dir: Path) -> dict:
    name = model["name"]
    pkg = model["package"]
    mod_name = model["module"]
    fam = model["family"]
    t0 = time.time()

    try:
        full = f"torchbenchmark.{pkg}.{mod_name}"
        mod = importlib.import_module(full)
        m = getattr(mod, "Model")(test="eval", device="cpu", batch_size=None)
        module, example_inputs = m.get_module()
        module.eval()
        module = _apply_model_patches(name, module)

        kwarg_names = None
        if isinstance(example_inputs, dict):
            kwarg_names = list(example_inputs.keys())
            args_tuple = tuple(example_inputs.values())
        elif isinstance(example_inputs, (tuple, list)):
            args_tuple = tuple(example_inputs)
        else:
            args_tuple = (example_inputs,)

        _needs_wrapper = fam == "transformers" or kwarg_names is not None or name in (
            "doctr_det_predictor", "doctr_reco_predictor"
        )
        if _needs_wrapper:
            module = TensorOutputWrapper(module, kwarg_names=kwarg_names).eval()

        with torch.no_grad():
            edge_model = litert_torch.convert(module, args_tuple, strict_export=False)

        out_path = out_dir / f"{name}.tflite"
        edge_model.export(str(out_path))

        size_mb = out_path.stat().st_size / (1024 * 1024)
        return dict(name=name, family=fam, status="PASS",
                    elapsed_s=f"{time.time()-t0:.1f}",
                    out_path=str(out_path), error=f"{size_mb:.1f}MB")

    except Exception as e:
        err = f"{type(e).__name__}: {e}".splitlines()[0][:300]
        traceback.print_exc(limit=5)
        return dict(name=name, family=fam, status="FAIL",
                    elapsed_s=f"{time.time()-t0:.1f}", out_path="", error=err)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    models = load_manifest(args.manifest)
    prior = load_csv_status(STATUS_CSV)
    pt2_pass = {name for name, st in load_csv_status(PT2_STATUS_CSV).items() if st == "PASS"}
    selected = filter_models(models, args, prior, pt2_pass)

    print(f"litert_torch:   {litert_torch.__version__}")
    print(f"torch:          {torch.__version__}")
    print(f"Output dir:     {args.out_dir}")
    print(f"pt2 PASS pool:  {len(pt2_pass)}")
    print(f"To process:     {len(selected)}")

    if args.list_only:
        for m in selected:
            print(f"  {m['family']:15s}  {m['name']}")
        return 0

    by_name = {m["name"]: m for m in models}
    rows_by_name: dict[str, dict] = {
        name: {"name": name, "family": by_name[name]["family"], "status": st,
               "elapsed_s": "", "out_path": "", "error": ""}
        for name, st in prior.items() if name in by_name
    }

    for i, m in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {m['name']} ({m['family']})")
        row = export_one(m, args.out_dir)
        rows_by_name[row["name"]] = row
        print(f"  -> {row['status']}  ({row['elapsed_s']}s)  {row['error']}")
        write_status(STATUS_CSV, [rows_by_name[n] for n in (r["name"] for r in models) if n in rows_by_name])

    counts: dict[str, int] = {}
    for r in rows_by_name.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n=== Summary ===")
    for k in ("PASS", "FAIL", "SKIP", "TIMEOUT"):
        print(f"  {k:8s} {counts.get(k, 0)}")
    print(f"  TOTAL   {sum(counts.values())}")
    return 0 if counts.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
