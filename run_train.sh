#!/bin/bash

case_num=1
GPUs="1,2"
batch_size=32

BASE_DIR="$HOME/Emanations/DLPitchEstimation/Emanation"
DATA_DIR="$BASE_DIR/programs/Venkatesh/results_fulldataset_Apr30th"

pickle_file="$DATA_DIR/iq_dict_case${case_num}_nodec.pkl"
SESSION="train_case${case_num}"
LOG_FILE="train_case${case_num}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
fi

tmux new -d -s "$SESSION" "
source ~/miniforge3/etc/profile.d/conda.sh;
conda activate April17th;
export CUDA_VISIBLE_DEVICES=$GPUs;
cd $BASE_DIR;
python programs/core/trainModel.py \
  --case $case_num \
  --data_path $pickle_file \
  --multi_gpu \
  --batch_size $batch_size \
  | tee $LOG_FILE
"
