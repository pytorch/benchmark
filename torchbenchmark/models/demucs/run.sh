#! /bin/bash
set -e

if [ -f metadata/musdb.json ]; then
  rm metadata/musdb.json
fi

for f in checkpoints evals logs models; do
    if [ -d $f ]; then
        rm -r $f
    fi
done

python3 -m demucs --musdb "$(pwd)/sample_data/" \
    --batch_size 1 \
    --device cuda \
    --workers 1 \
    --eval_workers 1 \
    --restart \
    --remix_group_size 1 \
    --samples 100000 \
    --repeat 1 \
    --epochs 1 \
    "$@"
