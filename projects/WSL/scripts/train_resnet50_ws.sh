#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

array=( $@ )
len=${#array[@]}
ARGS=${array[@]:0:$len}

EXP_DIR="output/resnet50_ws_`date +'%Y-%m-%d_%H-%M-%S'`"

mkdir -p "${EXP_DIR}"
LOG="${EXP_DIR}/train.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


echo ---------------------------------------------------------------------
git log -1
git submodule foreach 'git log -1'
echo ---------------------------------------------------------------------

python3 projects/WSL/tools/imagenet.py \
	-a resnet50_ws \
	--dist-url 'tcp://127.0.0.1:12345' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1 \
	--rank 0 \
	--output_dir ${EXP_DIR} \
	--epochs 120 \
	--batch-size 128 \
	--learning-rate 0.05 \
	--workers 12 \
	--print-freq 100 \
	--weights 'detectron2://ImageNetPretrained/MSRA/R-50.pkl' \
	${ARGS}
