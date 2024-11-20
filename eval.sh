#!/bin/bash
#python evaluate.py --model=weights/FlowDiffuser-things.pth  --dataset=sintel
#python evaluate.py --model=weights/FlowDiffuser-things.pth  --dataset=kitti


GPU_ID=$1

DATA_EVAL="kitti"
DATA_EVAL="sintel"
DATA_DIR="/home/ccj/code/tpk2scene/datasets"

MACHINE_NAME="rtxa6ks3"

flag=true
#flag=false
if [ "$flag" = true ]; then
    CKPT_PTH="/home/ccj/code/tpk2scene/checkpoints_nfs/pretrained/flowDiffuser/FlowDiffuser-things.pth"
    if [ $DATA_EVAL = "kitti" ]; then
        RESULT_DIR="/home/ccj/code/tpk2scene/results_nfs/flowdiffuser/kt15"
    elif [ $DATA_EVAL = "sintel" ]; then
        RESULT_DIR="/home/ccj/code/tpk2scene/results_nfs/flowdiffuser/sintel"
    fi
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate.py \
        --model $CKPT_PTH \
        --dataset $DATA_EVAL \
        --data_dir $DATA_DIR \
        --result_dir $RESULT_DIR \
        --machine_name $MACHINE_NAME \
        --eval_gpu_id $GPU_ID
fi