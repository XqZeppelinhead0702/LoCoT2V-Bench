EVAL_PATH=${1:-"./eval_videos/example.json"}
RESULT_DIR=${2:-"./results/example/"}

mkdir -p $RESULT_DIR

conda activate locot2v_bench

python ./locot2v_eval/eval_suite/eval_herd.py \
    --eval_data $EVAL_PATH \
    --mllm_path ./ckpts/Qwen3-VL-8B-Instruct \
    --src_path ./data/prompt.json \
    --result_path ${RESULT_DIR}/eval_herd.json