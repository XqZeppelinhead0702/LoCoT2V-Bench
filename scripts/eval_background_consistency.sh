EVAL_PATH=${1:-"./eval_videos/example.json"}
RESULT_DIR=${2:-"./results/example/"}

mkdir -p $RESULT_DIR

conda activate locot2v_bench

python ./locot2v_eval/eval_suite/eval_background_consistency.py \
    --model_name_or_path ./ckpts/fg-clip2-base \
    --eval_data $EVAL_PATH \
    --result_path ${RESULT_DIR}/eval_background_consistency.json