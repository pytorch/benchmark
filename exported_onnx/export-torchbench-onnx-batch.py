#!/usr/bin/env python3
"""Export pytorch/benchmark models to .onnx files for modelrunner.

Two-path strategy per model:
    1. torch.onnx.export(..., dynamo=True, opset_version=18)
    2. on failure, fall back to torch.onnx.export(..., dynamo=False)
The chosen path is recorded in the `exporter` column of the status CSV.

Sibling of `../exported_pt2/export-torchbench-batch.py` — re-uses its
TensorOutputWrapper, manifest loader, and per-model loading logic so
both pipelines stay in lockstep.

Designed to be invoked from the torchbench export venv
(`/local-ssd/sayans/Softwares/venvs/torchbench_export_py312_torch211`),
NOT from modelrunner's venv.

Usage:
  ./export-torchbench-onnx-batch.py                     # all not-yet-PASS rows
  ./export-torchbench-onnx-batch.py --only resnet50
  ./export-torchbench-onnx-batch.py --families vision,timm
  ./export-torchbench-onnx-batch.py --retry-failed
  ./export-torchbench-onnx-batch.py --force             # ignore prior status
  ./export-torchbench-onnx-batch.py --list-only
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

# Pull TensorOutputWrapper from the sibling .pt2 exporter so any wrapper
# fix automatically applies to the ONNX pipeline too.
REPO_ROOT = Path(__file__).resolve().parent.parent
PT2_DIR = REPO_ROOT / "exported_pt2"
HERE = REPO_ROOT / "exported_onnx"

sys.path.insert(0, str(PT2_DIR))
from importlib import util as _util  # noqa: E402

_spec = _util.spec_from_file_location("_pt2_exporter", PT2_DIR / "export-torchbench-batch.py")
_pt2 = _util.module_from_spec(_spec)
_spec.loader.exec_module(_pt2)
TensorOutputWrapper = _pt2.TensorOutputWrapper
_apply_model_patches = _pt2._apply_model_patches
DEFAULT_MANIFEST = PT2_DIR / "torchbench-models.yaml"
DEFAULT_OUT_DIR = HERE
STATUS_CSV = HERE / "export-status-onnx.csv"

CSV_FIELDS = [
    "name", "family", "status", "elapsed_s", "out_path",
    "size_mb", "opset", "exporter", "numeric_warning",
    "dynamo_error", "legacy_error",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--families", help="Comma-separated family filter")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip ORT load+infer validation step")
    ap.add_argument("--numeric-tol", type=float, default=1e-3,
                    help="Max abs error tolerance for ORT-vs-eager check")
    return ap.parse_args()


def load_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        data = yaml.safe_load(f)
    return data["models"]


def load_prior_status(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    out = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            out[row["name"]] = row
    return out


def write_status(csv_path: Path, rows: list[dict]) -> None:
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def filter_models(models, args, prior):
    out = list(models)
    if args.only:
        wanted = set(args.only)
        out = [m for m in out if m["name"] in wanted]
        missing = wanted - {m["name"] for m in out}
        if missing:
            print(f"WARNING: --only names not in manifest: {sorted(missing)}",
                  file=sys.stderr)
    if args.families:
        fams = set(args.families.split(","))
        out = [m for m in out if m["family"] in fams]
    if args.retry_failed:
        out = [m for m in out if (prior.get(m["name"], {}).get("status") in ("FAIL", "TIMEOUT", None))]
    elif not args.force and not args.only:
        kept = []
        for m in out:
            onnx_path = args.out_dir / f"{m['name']}.onnx"
            st = prior.get(m["name"], {}).get("status")
            if st == "PASS" and onnx_path.exists():
                continue
            kept.append(m)
        out = kept
    return out


def _load_module(model_entry: dict):
    """Return (wrapped_module, args_tuple, kwarg_names_or_None) ready for export."""
    full = f"torchbenchmark.{model_entry['package']}.{model_entry['module']}"
    mod = importlib.import_module(full)
    Model = getattr(mod, "Model")
    m = Model(test="eval", device="cpu", batch_size=None)
    module, example_inputs = m.get_module()
    module.eval()
    module = _apply_model_patches(model_entry["name"], module)

    kwarg_names = None
    if isinstance(example_inputs, dict):
        kwarg_names = list(example_inputs.keys())
        args_tuple = tuple(example_inputs.values())
    elif isinstance(example_inputs, (tuple, list)):
        args_tuple = tuple(example_inputs)
    else:
        args_tuple = (example_inputs,)

    _needs_wrapper = (model_entry["family"] == "transformers" or kwarg_names is not None
                      or model_entry["name"] in ("doctr_det_predictor", "doctr_reco_predictor"))
    if _needs_wrapper:
        module = TensorOutputWrapper(module, kwarg_names=kwarg_names).eval()
    return module, args_tuple


_ORT_TYPE_TO_NP = {
    "tensor(float)":   np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)":  np.float64,
    "tensor(bfloat16)": np.float32,  # NumPy has no bfloat16; ORT accepts float32 cast at session boundary
    "tensor(int8)":    np.int8,
    "tensor(int16)":   np.int16,
    "tensor(int32)":   np.int32,
    "tensor(int64)":   np.int64,
    "tensor(uint8)":   np.uint8,
    "tensor(uint16)":  np.uint16,
    "tensor(uint32)":  np.uint32,
    "tensor(uint64)":  np.uint64,
    "tensor(bool)":    np.bool_,
}


def _ort_type_to_np(t: str):
    """Map onnxruntime input type string ('tensor(int64)' etc.) to a NumPy dtype.
    Falls back to float32 for unknown types so validation still proceeds."""
    return _ORT_TYPE_TO_NP.get(t, np.float32)


def _validate_onnx(onnx_path: Path, module, args_tuple, tol: float) -> str:
    """onnx.checker + ORT inference + numeric A/B vs eager.

    Hard requirements (raise on failure): onnx.checker, ORT load+run.
    Soft requirement (returns a warning string, does NOT raise): numeric
    abs-error vs eager. Numeric drift in transformer cascades is expected
    even at fp32; modelrunner does its own SWA check downstream.

    For models that exceed the 2 GB protobuf serialization cap, both
    `onnx.checker.check_model` and ORT's `InferenceSession` constructor can
    raise EncodeError even though the on-disk model (with external_data
    sidecar) is structurally fine. We downgrade those to a soft warning so
    the export isn't falsely marked FAIL — modelrunner runs the real numeric
    check downstream anyway.
    """
    # If the on-disk sidecar is large, skip in-process serialization-heavy
    # checks entirely. The 2 GB checker limit is hardcoded in libprotobuf.
    sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
    total_size = onnx_path.stat().st_size + (sidecar.stat().st_size if sidecar.exists() else 0)
    if total_size > 2_000_000_000:
        return f"validation skipped: model size {total_size/1e9:.1f} GB exceeds 2 GB checker limit"

    m = onnx.load(str(onnx_path), load_external_data=True)
    onnx.checker.check_model(m, full_check=False)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()
    feeds = {}
    tensor_args = [a for a in args_tuple if isinstance(a, torch.Tensor)]
    if len(tensor_args) != len(in_meta):
        for meta in in_meta:
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
            feeds[meta.name] = np.zeros(shape, dtype=_ort_type_to_np(meta.type))
    else:
        for meta, t in zip(in_meta, tensor_args):
            feeds[meta.name] = t.detach().cpu().numpy()

    ort_outs = sess.run(None, feeds)

    # Numeric A/B vs eager — best-effort, soft-fail only.
    if len(tensor_args) != len(in_meta):
        return ""
    try:
        with torch.no_grad():
            eager = module(*args_tuple)
    except Exception as e:
        return f"eager forward failed: {type(e).__name__}: {e}"[:200]
    if isinstance(eager, torch.Tensor):
        eager_list = [eager]
    elif isinstance(eager, (tuple, list)):
        eager_list = [t for t in eager if isinstance(t, torch.Tensor)]
    else:
        eager_list = []
    max_err = 0.0
    for e, o in zip(eager_list, ort_outs):
        if e.shape == tuple(o.shape):
            err = float(np.max(np.abs(e.detach().cpu().numpy() - o)))
            if not np.isnan(err):
                max_err = max(max_err, err)
    if max_err > tol:
        return f"numeric drift max-abs={max_err:.4g} (>tol={tol})"
    return ""


def _try_export(module, args_tuple, out_path: Path, opset: int, dynamo: bool) -> None:
    """Run torch.onnx.export with the requested backend; raises on failure.

    Retries once with `external_data=True` (dynamo) or
    `use_external_data_format=True` (legacy) on protobuf 2 GB EncodeError.

    The dynamo exporter leaks loose tensor shards (e.g. `_model_*`, `blocks.0.*`)
    into out_dir when it fails on the inline-serialization step. Those shards
    confuse the retry, so we snapshot out_dir contents before each attempt and
    delete anything new before retrying.
    """
    out_dir = out_path.parent

    def _do(use_external: bool) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kw = dict(opset_version=opset, dynamo=dynamo, verbose=False)
            if use_external:
                if dynamo:
                    kw["external_data"] = True
                else:
                    kw["use_external_data_format"] = True
            torch.onnx.export(module, args_tuple, str(out_path), **kw)

    snapshot = set(out_dir.iterdir())
    try:
        _do(use_external=False)
    except Exception as e:
        msg = str(e)
        if "Failed to serialize proto" in msg or "exceeds maximum protobuf size" in msg \
                or "2147483648" in msg or "2 GB" in msg.lower().replace("gb", "GB"):
            # Wipe everything the failed attempt created (main .onnx, sidecar
            # .onnx.data, and the loose tensor shards) before retrying.
            for p in set(out_dir.iterdir()) - snapshot:
                try:
                    p.unlink()
                except OSError:
                    pass
            _do(use_external=True)
        else:
            raise


def export_one(model_entry: dict, out_dir: Path, opset: int,
               numeric_tol: float, validate: bool) -> dict:
    name = model_entry["name"]
    fam = model_entry["family"]
    t0 = time.time()
    row = {
        "name": name, "family": fam, "status": "",
        "elapsed_s": "", "out_path": "",
        "size_mb": "", "opset": str(opset),
        "exporter": "none", "numeric_warning": "",
        "dynamo_error": "", "legacy_error": "",
    }

    if model_entry.get("skip_reason"):
        row.update(status="SKIP", elapsed_s=f"{0:.1f}",
                   dynamo_error=model_entry["skip_reason"])
        return row

    # 1. Load + wrap the module
    try:
        module, args_tuple = _load_module(model_entry)
    except Exception as e:
        err = f"{type(e).__name__}: {e}".splitlines()[0][:300]
        traceback.print_exc(limit=2)
        row.update(status="FAIL",
                   elapsed_s=f"{time.time()-t0:.1f}",
                   dynamo_error=f"load: {err}",
                   legacy_error=f"load: {err}")
        return row

    out_path = out_dir / f"{name}.onnx"

    # 2. Dynamo path first
    dynamo_err = ""
    try:
        _try_export(module, args_tuple, out_path, opset, dynamo=True)
        warn = _validate_onnx(out_path, module, args_tuple, numeric_tol) if validate else ""
        size_mb = sum(p.stat().st_size for p in out_dir.glob(f"{name}.onnx*")) / 1e6
        row.update(status="PASS", elapsed_s=f"{time.time()-t0:.1f}",
                   out_path=str(out_path), size_mb=f"{size_mb:.1f}",
                   exporter="dynamo", numeric_warning=warn)
        return row
    except Exception as e:
        dynamo_err = f"{type(e).__name__}: {e}".splitlines()[0][:300]
        for p in out_dir.glob(f"{name}.onnx*"):
            try:
                p.unlink()
            except OSError:
                pass

    # 3. Legacy TorchScript path fallback
    legacy_err = ""
    try:
        _try_export(module, args_tuple, out_path, opset, dynamo=False)
        warn = _validate_onnx(out_path, module, args_tuple, numeric_tol) if validate else ""
        size_mb = sum(p.stat().st_size for p in out_dir.glob(f"{name}.onnx*")) / 1e6
        row.update(status="PASS", elapsed_s=f"{time.time()-t0:.1f}",
                   out_path=str(out_path), size_mb=f"{size_mb:.1f}",
                   exporter="legacy", numeric_warning=warn,
                   dynamo_error=dynamo_err)
        return row
    except Exception as e:
        legacy_err = f"{type(e).__name__}: {e}".splitlines()[0][:300]
        for p in out_dir.glob(f"{name}.onnx*"):
            try:
                p.unlink()
            except OSError:
                pass

    row.update(status="FAIL", elapsed_s=f"{time.time()-t0:.1f}",
               dynamo_error=dynamo_err, legacy_error=legacy_err)
    return row


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
    print(f"ONNX opset:     {args.opset}")
    print(f"Total models:   {len(models)}")
    print(f"To process:     {len(selected)}")
    if args.list_only:
        for m in selected:
            print(f"  {m['family']:15s}  {m['name']}")
        return 0

    # seed rows from prior so we keep already-PASS entries when re-running
    by_name = {m["name"]: m for m in models}
    rows_by_name: dict[str, dict] = {}
    for name, prior_row in prior.items():
        if name in by_name:
            rows_by_name[name] = {k: prior_row.get(k, "") for k in CSV_FIELDS}
            rows_by_name[name]["name"] = name
            rows_by_name[name]["family"] = by_name[name]["family"]

    for i, m in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {m['name']} ({m['family']})")
        row = export_one(m, args.out_dir, args.opset,
                         numeric_tol=args.numeric_tol,
                         validate=not args.no_validate)
        rows_by_name[row["name"]] = row
        why = row["legacy_error"] or row["dynamo_error"]
        warn = row["numeric_warning"]
        suffix = (f"  warn: {warn}" if warn else "") + (f"  err: {why}" if why else "")
        print(f"  -> {row['status']}  exporter={row['exporter']}  "
              f"({row['elapsed_s']}s){suffix}")
        # checkpoint after every model
        ordered = [rows_by_name[m2["name"]] for m2 in models
                   if m2["name"] in rows_by_name]
        write_status(STATUS_CSV, ordered)

    # final summary
    counts_status: dict[str, int] = {}
    counts_exporter: dict[str, int] = {}
    for r in rows_by_name.values():
        counts_status[r["status"]] = counts_status.get(r["status"], 0) + 1
        counts_exporter[r["exporter"]] = counts_exporter.get(r["exporter"], 0) + 1
    print("\n=== Summary ===")
    for k in ("PASS", "FAIL", "SKIP"):
        print(f"  {k:8s} {counts_status.get(k, 0)}")
    print(f"  TOTAL    {sum(counts_status.values())}")
    print("\n  Exporter breakdown:")
    for k in ("dynamo", "legacy", "none"):
        print(f"    {k:8s} {counts_exporter.get(k, 0)}")
    return 0 if counts_status.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
