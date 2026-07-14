EVAL_PATH=${1:-"./eval_videos/example.json"}
RESULT_DIR=${2:-"./results/example/"}

mkdir -p $RESULT_DIR

conda activate pq_eval

python ./locot2v_eval/eval_suite/eval_perceptual_quality.py \
    --model_name_or_path ./ckpts/DeQA-Score-Mix3 \
    --eval_data $EVAL_PATH \
    --result_path ${RESULT_DIR}/eval_perceptual_quality.json