#!/usr/bin/env python3
"""Export pytorch/benchmark models to .pt2 ExportedProgram files.

Data-driven from torchbench-models.yaml.  Writes export-status.csv
alongside the .pt2 outputs so future re-runs can target only failures.

Designed to be invoked from the torchbench export venv (PyTorch 2.11+),
NOT from modelrunner's venv.

Usage:
  ./export-torchbench-batch.py                       # export everything not yet PASSing
  ./export-torchbench-batch.py --only resnet50       # one model
  ./export-torchbench-batch.py --families vision,timm
  ./export-torchbench-batch.py --retry-failed        # only re-run FAIL rows from previous CSV
  ./export-torchbench-batch.py --force               # ignore prior status, re-export all
  ./export-torchbench-batch.py --list-only           # print what would run, no work
  ./export-torchbench-batch.py --install-deps        # pip install <model>/requirements.txt first
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import yaml


class TensorOutputWrapper(nn.Module):
    """Unwrap HuggingFace dataclass outputs to plain tensors so .pt2
    round-trips through torch.export.load. Without this, transformers'
    CausalLMOutputWithPast / Seq2SeqLMOutput / etc. fail to deserialize
    even when `transformers` is imported in the loader.

    Also converts dict-style example_inputs into positional args so the
    exported program has a positional signature (modelrunner only feeds
    positional args)."""

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

def _apply_model_patches(name: str, module: nn.Module) -> nn.Module:
    """Per-model patches applied after get_module() and before torch.export.export."""
    if name in ("doctr_det_predictor", "doctr_reco_predictor"):
        # DBNet / CRNN forward calls .numpy() or data-dependent CTC postprocessing
        # when exportable=False. The exportable flag returns only logits, skipping
        # those paths.
        if hasattr(module, "exportable"):
            module.exportable = True
    return module


REPO_ROOT = Path(__file__).resolve().parent.parent
HERE = REPO_ROOT / "exported_pt2"
DEFAULT_MANIFEST = HERE / "torchbench-models.yaml"
DEFAULT_OUT_DIR = HERE
STATUS_CSV = HERE / "export-status.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--only", action="append", default=[], help="Model name(s) to export (repeatable).")
    ap.add_argument("--families", help="Comma-separated family filter, e.g. vision,timm")
    ap.add_argument("--retry-failed", action="store_true", help="Re-run only FAIL rows from existing CSV.")
    ap.add_argument("--force", action="store_true", help="Re-export even if a .pt2 + PASS row already exists.")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--install-deps", action="store_true",
                    help="Run 'pip install -r requirements.txt' for each model's dir before export.")
    ap.add_argument("--timeout", type=int, default=900,
                    help="Per-model wall-clock timeout in seconds (default 900).")
    return ap.parse_args()


def load_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        data = yaml.safe_load(f)
    models = data["models"]
    seen = set()
    for m in models:
        if m["name"] in seen:
            raise RuntimeError(f"Duplicate entry in manifest: {m['name']}")
        seen.add(m["name"])
    return models


def load_prior_status(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    out = {}
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["name"]] = row["status"]
    return out


def write_status(csv_path: Path, rows: list[dict]) -> None:
    fields = ["name", "family", "status", "elapsed_s", "out_path", "error"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def filter_models(models: list[dict], args: argparse.Namespace, prior: dict[str, str]) -> list[dict]:
    out = list(models)
    if args.only:
        wanted = set(args.only)
        out = [m for m in out if m["name"] in wanted]
        missing = wanted - {m["name"] for m in out}
        if missing:
            print(f"WARNING: --only names not in manifest: {sorted(missing)}", file=sys.stderr)
    if args.families:
        fams = set(args.families.split(","))
        out = [m for m in out if m["family"] in fams]
    if args.retry_failed:
        out = [m for m in out if prior.get(m["name"]) in ("FAIL", "TIMEOUT", None)]
    elif not args.force and not args.only:
        # default: skip models that already PASSed and whose .pt2 still exists
        kept = []
        for m in out:
            pt2 = args.out_dir / f"{m['name']}.pt2"
            if prior.get(m["name"]) == "PASS" and pt2.exists():
                continue
            kept.append(m)
        out = kept
    return out


def install_model_deps(model: dict) -> None:
    """Run pip install -r requirements.txt if the model dir has one."""
    req = REPO_ROOT / "torchbenchmark" / model["package"] / model["module"] / "requirements.txt"
    if not req.exists():
        return
    print(f"  [deps] installing {req}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=False)


def export_one(model: dict, out_dir: Path) -> dict:
    """Instantiate the torchbench Model, export to .pt2, validate."""
    name = model["name"]
    pkg = model["package"]
    mod_name = model["module"]
    fam = model["family"]
    t0 = time.time()

    if model.get("skip_reason"):
        return dict(name=name, family=fam, status="SKIP",
                    elapsed_s=f"{0:.1f}", out_path="", error=model["skip_reason"])

    try:
        full = f"torchbenchmark.{pkg}.{mod_name}"
        mod = importlib.import_module(full)
        Model = getattr(mod, "Model")
        m = Model(test="eval", device="cpu", batch_size=None)
        module, example_inputs = m.get_module()
        module = _apply_model_patches(name, module)
        module.eval()

        # Determine positional/keyword input layout.
        kwarg_names = None
        if isinstance(example_inputs, dict):
            kwarg_names = list(example_inputs.keys())
            args_tuple = tuple(example_inputs.values())
            kwargs = {}
        elif isinstance(example_inputs, (tuple, list)):
            args_tuple = tuple(example_inputs)
            kwargs = {}
        else:
            args_tuple = (example_inputs,)
            kwargs = {}

        # HF dataclass outputs (CausalLMOutputWithPast etc.) don't survive
        # torch.export round-trip — flatten them to plain tensors before export.
        # Wrapper also forces a positional signature for dict-style inputs so
        # modelrunner can drive the exported program.
        # doctr models with exportable=True return {"logits": tensor} dicts;
        # wrap them the same way as transformers families.
        _needs_wrapper = fam == "transformers" or kwarg_names is not None or name in (
            "doctr_det_predictor", "doctr_reco_predictor"
        )
        if _needs_wrapper:
            module = TensorOutputWrapper(module, kwarg_names=kwarg_names).eval()

        with torch.no_grad():
            ep = torch.export.export(module, args_tuple, kwargs=kwargs, strict=False)

        # torch 2.11 bug: vmap/functorch predispatch nodes (_add_batch_dim etc.)
        # are not serializable. run_decompositions({}) lowers them out; then
        # eliminate_dead_code removes any remaining dead nodes.
        ep = ep.run_decompositions({})
        ep.graph.eliminate_dead_code()

        out_path = out_dir / f"{name}.pt2"
        torch.export.save(ep, str(out_path))

        # round-trip validate
        loaded = torch.export.load(str(out_path))
        with torch.no_grad():
            _ = loaded.module()(*args_tuple, **kwargs)

        return dict(name=name, family=fam, status="PASS",
                    elapsed_s=f"{time.time()-t0:.1f}", out_path=str(out_path), error="")

    except Exception as e:
        err = f"{type(e).__name__}: {e}".splitlines()[0][:300]
        traceback.print_exc(limit=3)
        return dict(name=name, family=fam, status="FAIL",
                    elapsed_s=f"{time.time()-t0:.1f}", out_path="", error=err)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = load_manifest(args.manifest)
    prior = load_prior_status(STATUS_CSV)
    selected = filter_models(models, args, prior)

    print(f"Manifest:       {args.manifest}")
    print(f"Output dir:     {args.out_dir}")
    print(f"Status CSV:     {STATUS_CSV}")
    print(f"Torch:          {torch.__version__}")
    print(f"Total models:   {len(models)}")
    print(f"To process:     {len(selected)}")
    if args.list_only:
        for m in selected:
            print(f"  {m['family']:15s}  {m['name']}")
        return 0

    # seed rows with prior status, then overwrite the ones we re-run
    by_name = {m["name"]: m for m in models}
    rows_by_name: dict[str, dict] = {
        name: {"name": name, "family": by_name[name]["family"], "status": st,
               "elapsed_s": "", "out_path": "", "error": ""}
        for name, st in prior.items() if name in by_name
    }

    for i, m in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {m['name']} ({m['family']})")
        if args.install_deps:
            install_model_deps(m)
        row = export_one(m, args.out_dir)
        rows_by_name[row["name"]] = row
        print(f"  -> {row['status']}  ({row['elapsed_s']}s)  {row['error']}")
        # checkpoint after every model so a crash mid-sweep doesn't lose progress
        write_status(STATUS_CSV, [rows_by_name[n] for n in (rn["name"] for rn in models) if n in rows_by_name])

    # final summary
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
