#!/usr/bin/env bash
# Run flatbuffer_translate + tflite-opt MW pipeline on all .tflite files.
# Reports PASS/FAIL per model and writes tosa-legalize-status.csv.

set -euo pipefail

TFLITE_DIR="/local-ssd/sayans/Softwares/pytorch-benchmark/exported_litert"
TOSA_CONVERTER="/local-ssd/sayans/Softwares/tosa-converter-for-tflite-repo/mw"
FB_TRANSLATE="$TOSA_CONVERTER/bazel-bin/tosa_converter_for_tflite/flatbuffer_translate/flatbuffer_translate"
TFLITE_OPT="$TOSA_CONVERTER/bazel-bin/tosa_converter_for_tflite/tensor_passes/tflite-opt"
OUT_CSV="$TFLITE_DIR/tosa-legalize-status.csv"
TMP_DIR=$(mktemp -d /local-ssd/sayans/tmp.XXXXXX)

echo "name,status,error" > "$OUT_CSV"

pass=0
fail=0

for tflite in "$TFLITE_DIR"/*.tflite; do
    name=$(basename "$tflite" .tflite)
    mlir_file="$TMP_DIR/${name}.mlir"
    tosa_file="$TMP_DIR/${name}_tosa.mlir"

    printf "[%-40s] " "$name"

    # Step 1: flatbuffer -> MLIR
    if ! "$FB_TRANSLATE" --tflite-flatbuffer-to-mlir "$tflite" -o "$mlir_file" 2>/tmp/fb_err.txt; then
        err=$(head -1 /tmp/fb_err.txt | cut -c1-120)
        echo "FAIL (flatbuffer_translate: $err)"
        echo "$name,FAIL,flatbuffer_translate: $err" >> "$OUT_CSV"
        ((fail++)) || true
        continue
    fi

    # Step 2: TFL -> TOSA (MW pipeline)
    if ! "$TFLITE_OPT" --allow-unregistered-dialect \
        '--tfl-to-tensor-tosa-pipeline=dequantize-tfl-softmax=true' \
        "$mlir_file" -o "$tosa_file" 2>/tmp/tosa_err.txt; then
        err=$(grep -m1 'error:' /tmp/tosa_err.txt | sed 's/.*error: //' | cut -c1-120)
        echo "FAIL (tflite-opt: $err)"
        echo "$name,FAIL,tflite-opt: $err" >> "$OUT_CSV"
        ((fail++)) || true
        continue
    fi

    echo "PASS"
    echo "$name,PASS," >> "$OUT_CSV"
    ((pass++)) || true
done

rm -rf "$TMP_DIR"

echo ""
echo "=== Summary ==="
echo "  PASS  $pass"
echo "  FAIL  $fail"
echo "  TOTAL $((pass + fail))"
echo ""
echo "Results: $OUT_CSV"
