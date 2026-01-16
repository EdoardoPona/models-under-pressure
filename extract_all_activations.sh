#!/usr/bin/env bash

# Exit on error
set -euo pipefail

# Model and layer configuration
MODEL="meta-llama/Llama-3.2-1B-Instruct"
LAYER=11

# Resolve DATA_DIR: prefer env var, otherwise read from .env (required).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${DATA_DIR:-}" ]]; then
  BASE_DATA_DIR="${DATA_DIR}"
else
  ENV_FILE="${PROJECT_ROOT}/.env"
  # Assume .env exists and contains a single DATA_DIR line
  env_line=$(grep -E '^DATA_DIR=' "${ENV_FILE}")
  env_val=${env_line#DATA_DIR=}
  # Strip simple surrounding quotes if present
  env_val="${env_val%\"}"; env_val="${env_val#\"}"
  env_val="${env_val%\'}"; env_val="${env_val#\'}"

  # Expand ~ and other shell-style expansions
  eval "env_val_expanded=${env_val}"
  if [[ "${env_val_expanded}" = /* ]]; then
    BASE_DATA_DIR="${env_val_expanded}"
  else
    BASE_DATA_DIR="${PROJECT_ROOT}/${env_val_expanded}"
  fi
fi

echo "Using DATA_DIR=${BASE_DATA_DIR}"

echo "=== Extracting activations for synthetic train/test (prompts_4x) ==="
uv run mup acts store \
  --model "${MODEL}" \
  --layers "${LAYER}" \
  --dataset "${BASE_DATA_DIR}/training/prompts_4x/train.jsonl"

uv run mup acts store \
  --model "${MODEL}" \
  --layers "${LAYER}" \
  --dataset "${BASE_DATA_DIR}/training/prompts_4x/test.jsonl"

echo "=== Extracting activations for dev eval datasets ==="
for f in "${BASE_DATA_DIR}"/evals/dev/*.jsonl; do
  if [[ -f "${f}" ]]; then
    echo "Processing dev dataset: ${f}"
    uv run mup acts store \
      --model "${MODEL}" \
      --layers "${LAYER}" \
      --dataset "${f}"
  fi
done

echo "=== Extracting activations for test eval datasets ==="
for f in "${BASE_DATA_DIR}"/evals/test/*.jsonl; do
  if [[ -f "${f}" ]]; then
    echo "Processing test dataset: ${f}"
    uv run mup acts store \
      --model "${MODEL}" \
      --layers "${LAYER}" \
      --dataset "${f}"
  fi
done

echo "All activations extracted. You can now run evaluate_probe with compute_activations=false."
