VIDEO_DIR="path/to/your/videos"
EVAL_PATH="path/to/your/eval_videos.json"
RESULT_DIR="path/to/save/your/results"

# Get video list
python ./scripts/make_eval_list.py $VIDEO_DIR $EVAL_PATH

# Evaluate your videos on all of our metrics
bash ./scripts/eval_perceptual_quality.sh $EVAL_PATH $RESULT_DIR
bash ./scripts/eval_overall_alignment.sh $EVAL_PATH $RESULT_DIR
bash ./scripts/eval_fine_grained_alignment.sh $EVAL_PATH $RESULT_DIR
bash ./scripts/eval_character_consistency.sh $EVAL_PATH $RESULT_DIR
bash ./scripts/eval_background_consistency.sh $EVAL_PATH $RESULT_DIR
bash ./scripts/eval_warping_error.sh $EVAL_PATH $RESULT_DIR
bash ./scripts/eval_herd.sh $EVAL_PATH $RESULT_DIR
bash ./scripts/eval_dynamic_quality.sh $EVAL_PATH $RESULT_DIR