# Exporting TorchBench Models

Export all 110 commercially-licensed torchbench models to `.pt2` (TorchExport), `.onnx`, and `.tflite` (LiteRT) formats.

## Prerequisites

- Python 3.12+
- CUDA-capable GPU (for model instantiation)
- ~50 GB disk for exported artifacts

## Setup

```bash
# Clone and enter the repo
git clone https://github.com/sahas3/benchmark.git
cd benchmark

# Create a venv
python -m venv .venv
source .venv/bin/activate

# Install torchbench (editable, pulls model dependencies)
pip install -e .

# Install export dependencies
pip install -r requirements-export.txt
```

### Optional: LiteRT export

`litert-torch` requires a separate install and only supports Linux x86_64:

```bash
pip install litert-torch
```

If not installed, the unified wrapper skips `.tflite` export automatically.

### Installing model weights

Most models download weights on first use. To pre-download all:

```bash
python install.py
```

This can take a while and requires ~30 GB for weights.

## Usage

### Unified wrapper (recommended)

```bash
# Export all formats (skips models already marked PASS)
./export-all.sh

# Export specific formats
./export-all.sh pt2 onnx
./export-all.sh litert

# Single model
./export-all.sh --only resnet50

# Filter by family (vision, nlp, gnn, timm, etc.)
./export-all.sh --families vision

# Re-export everything (ignore prior status)
./export-all.sh --force
```

### Individual scripts

```bash
# PT2 export
python exported_pt2/export-torchbench-batch.py

# ONNX export (gates on pt2 PASS status)
python exported_onnx/export-torchbench-onnx-batch.py

# LiteRT export (gates on pt2 PASS status)
python exported_litert/export-torchbench-litert-batch.py

# TOSA legalization check (requires flatbuffer_translate + tflite-opt)
./exported_litert/tosa-legalize-batch.sh
```

All scripts support `--only MODEL`, `--families FAMILY`, and `--force`.

## Output structure

```
exported_pt2/
├── export-status.csv          # model,status,error columns
├── torchbench-models.yaml     # model manifest (edit to add/skip models)
├── *.pt2                      # exported artifacts (gitignored)
exported_onnx/
├── export-status-onnx.csv
├── *.onnx                     # (gitignored)
exported_litert/
├── export-status.csv
├── tosa-legalize-status.csv
├── *.tflite                   # (gitignored)
```

## Model manifest

The export is data-driven from `exported_pt2/torchbench-models.yaml`. Each entry specifies:

| Field | Description |
|-------|-------------|
| `name` | Model name (also the output filename stem) |
| `package` | `models` or `canary_models` |
| `module` | Subdirectory name under `torchbenchmark/` |
| `family` | Grouping: vision, nlp, gnn, timm, speech, generative, misc |
| `skip_reason` | If set, exporter emits SKIP without attempting export |

To add a model, append an entry. To skip one, add `skip_reason: "reason"`.

## Export pipeline

```
torchbench model
    │
    ▼
torch.export.export(strict=False)
    │
    ▼
ep.run_decompositions({})    ← fixes _add_batch_dim serialization
    │
    ▼
torch.export.save() → .pt2
    │
    ├──► torch.onnx.export() → .onnx
    │
    └──► litert_torch.convert() → .tflite
              │
              ▼
         tflite-opt --tfl-to-tensor-tosa-pipeline → TOSA legalization check
```

## Current results

| Format | Pass | Fail | Skip | Total |
|--------|------|------|------|-------|
| .pt2 | 62 | 15 | 33 | 110 |
| .onnx | 64 | 12 | 34 | 110 |
| .tflite | 53 | 9 | — | 62 (pt2-PASS pool) |
| TOSA legalize | 53 | 0 | — | 53 (.tflite pool) |

## Known limitations

### LiteRT export failures (9 models)

| Cause | Models |
|-------|--------|
| fp16/bf16 unsupported | hf_distil_whisper, hf_Whisper, stable_diffusion_xl, phi_2 |
| float64 ops | pytorch_stargan |
| transposed conv output_padding | drq, soft_actor_critic |
| EmbeddingBag | dlrm |
| empty_permuted | lennard_jones |

### Skip reasons (common)

- `dynamic shapes only` — model requires symbolic dimensions not supported by static export
- `no eval mode` — training-only model (e.g., detectron2, moco)
- `timeout / OOM` — model too large for single-GPU export

## Troubleshooting

**Model fails to load**: Run `python -c "from torchbenchmark.models.MODEL import Model; m = Model('eval', 'cuda')"` to isolate loading errors.

**OOM during export**: Try with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or on a GPU with more VRAM.

**ONNX export hangs**: Some large models (llama2, phi_2) take 10+ minutes. The script has no timeout — monitor with `nvidia-smi`.

**LiteRT "not supported" error**: Check if the failing op is in the [LiteRT supported ops list](https://ai.google.dev/edge/litert/models/ops_compatibility). Mixed-precision models often hit this.
