EVAL_PATH=${1:-"./eval_videos/example.json"}
RESULT_DIR=${2:-"./results/example/"}

mkdir -p $RESULT_DIR

conda activate vbench_env

python ./locot2v_eval/eval_suite/eval_warping_error.py \
    --eval_data $EVAL_PATH \
    --raft_path ./ckpts/RAFT/raft-things.pth \
    --result_path ${RESULT_DIR}/eval_warping_error.json

