MODEL="meta-llama/Llama-3.2-1B-Instruct"
LAYER=11

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

DATASETS = [ 
    "${BASE_DATA_DIR}/training/prompts_4x/train.jsonl",
    "${BASE_DATA_DIR}/training/prompts_4x/test.jsonl",
    "${BASE_DATA_DIR}/evals/dev/anthropic_balanced_apr_23.jsonl",
    "${BASE_DATA_DIR}/evals/test/anthropic_test_balanced_apr_23.jsonl"
]

for DATASET in ${DATASETS[@]}; do
    echo "Processing dataset: ${DATASET}"
    uv run mup acts store --model $MODEL --layer $LAYER --dataset $DATASET
done

echo "=== Finished extracting minimal activations ==="
