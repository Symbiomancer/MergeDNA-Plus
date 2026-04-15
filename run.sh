#!/bin/bash
# Train MergeDNA
# Reads args from config.json, CLI args override.
#
# Usage:
#   ./run.sh                              # use config.json
#   ./run.sh --merge_r 8 --dim 128        # override specific fields

source ~/diffusion/.venv/bin/activate

CONFIG=${CONFIG:-config.json}

ARGS=$(python -c "
import json
with open('$CONFIG') as f:
    cfg = json.load(f)
args = []
for k, v in cfg.items():
    if isinstance(v, bool):
        if v:
            args.append(f'--{k}')
    else:
        args.append(f'--{k}')
        args.append(str(v))
print(' '.join(args))
")

python train.py $ARGS "$@"
