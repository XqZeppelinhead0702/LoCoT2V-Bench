EVAL_PATH=${1:-"./eval_videos/example.json"}
RESULT_DIR=${2:-"./results/example/"}

mkdir -p $RESULT_DIR

conda activate vbench_env

python ./locot2v_eval/eval_suite/eval_dynamic_degree.py \
    --eval_data $EVAL_PATH \
    --raft_path ./ckpts/RAFT/raft-things.pth \
    --result_path ${RESULT_DIR}/eval_dynamic_degree.json

python ./locot2v_eval/eval_suite/eval_motion_smoothness.py \
    --eval_data $EVAL_PATH \
    --amt_path ./ckpts/AMT/amt-s.pth \
    --amt_config ./ckpts/AMT/AMT-S.yaml \
    --result_path ${RESULT_DIR}/eval_motion_smoothness.json


conda activate locot2v_bench

python ./locot2v_eval/eval_suite/eval_devil_scores.py \
    --eval_data $EVAL_PATH \
    --result_path ${RESULT_DIR}/eval_devil_scores.json

# merge the score into the final dynamic quality score
python ./scripts/compute_dq_scores.py $RESULT_DIR