#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m mlx_lm lora \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --train \
  --data data \
  --iters "${ITERS:-200}" \
  --batch-size 2 \
  --grad-accumulation-steps 2 \
  --grad-checkpoint \
  --num-layers 16 \
  --learning-rate 5e-5 \
  --mask-prompt \
  --max-seq-length 768 \
  --steps-per-report 10 \
  --steps-per-eval 100 \
  --val-batches 10 \
  --save-every 100 \
  --adapter-path adapters
