EVAL_PATH=${1:-"./eval_videos/example.json"}
RESULT_DIR=${2:-"./results/example/"}

mkdir -p $RESULT_DIR

conda activate sam3_env

python ./locot2v_eval/eval_suite/eval_character_consistency.py \
    --sam3_path ./ckpts/sam3 \
    --mllm_path ./ckpts/Qwen3-VL-4B-Instruct \
    --clip_path ./ckpts/fg-clip2-base \
    --src_path ./data/prompt.json \
    --eval_data $EVAL_PATH \
    --result_path ${RESULT_DIR}/eval_character_consistency.json