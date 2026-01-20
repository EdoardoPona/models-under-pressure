MODEL="meta-llama/Llama-3.2-1B-Instruct"
LAYER=11

BASE_DATA_DIR= # paste here for now

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
