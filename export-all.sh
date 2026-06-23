#!/usr/bin/env bash
# Export all torchbench models to .pt2, .onnx, and .tflite in sequence.
#
# Prerequisites:
#   1. Install torchbenchmark: pip install -e .
#   2. Install export deps: pip install -r requirements-export.txt
#
# Usage:
#   ./export-all.sh                   # export all formats (skips already-PASS)
#   ./export-all.sh --force           # re-export everything
#   ./export-all.sh --only resnet50   # single model, all formats
#   ./export-all.sh --families vision  # family filter
#   ./export-all.sh pt2              # only .pt2
#   ./export-all.sh onnx litert      # only .onnx and .tflite
#
# Arguments are passed through to each sub-script. If the first argument(s)
# are format names (pt2, onnx, litert), only those formats run.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse format selectors from args
FORMATS=()
PASS_ARGS=()
for arg in "$@"; do
    case "$arg" in
        pt2|onnx|litert) FORMATS+=("$arg") ;;
        *) PASS_ARGS+=("$arg") ;;
    esac
done

# Default: all formats
if [ ${#FORMATS[@]} -eq 0 ]; then
    FORMATS=(pt2 onnx litert)
fi

run_pt2() {
    echo "===== Exporting to .pt2 ====="
    python "$SCRIPT_DIR/exported_pt2/export-torchbench-batch.py" "${PASS_ARGS[@]}"
}

run_onnx() {
    echo ""
    echo "===== Exporting to .onnx ====="
    python "$SCRIPT_DIR/exported_onnx/export-torchbench-onnx-batch.py" "${PASS_ARGS[@]}"
}

run_litert() {
    if ! python -c "import litert_torch" 2>/dev/null; then
        echo ""
        echo "===== Skipping .tflite (litert_torch not installed) ====="
        return 0
    fi
    echo ""
    echo "===== Exporting to .tflite ====="
    python "$SCRIPT_DIR/exported_litert/export-torchbench-litert-batch.py" "${PASS_ARGS[@]}"
}

for fmt in "${FORMATS[@]}"; do
    "run_$fmt"
done

echo ""
echo "===== Done ====="
echo "Results:"
for fmt in "${FORMATS[@]}"; do
    case "$fmt" in
        pt2)    echo "  .pt2:    $SCRIPT_DIR/exported_pt2/export-status.csv" ;;
        onnx)   echo "  .onnx:   $SCRIPT_DIR/exported_onnx/export-status-onnx.csv" ;;
        litert) echo "  .tflite: $SCRIPT_DIR/exported_litert/export-status.csv" ;;
    esac
done
