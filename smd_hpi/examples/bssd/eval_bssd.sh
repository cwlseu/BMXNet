#!/bin/sh

#PBS -N voc0712_bssd300_eval
#PBS -l nodes=gpu06
#PBS -l walltime=4:00:00
#PBD -d /home/limin/Repo/BMXNet


export CUDA_HOME=/soft/cuda7.5
export PATH=/soft/cuda7.5/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib64
export PYTHONPATH=/home/limin/Repo/BMXNet/python:/home/limin/Repo/caffe/python:$PYTHONPATH 
export ANACONDA_HOME=/soft/anaconda
export CAFFE_LIB=/home/limin/Repo/3rdparty/caffelib
export PATH=$CAFFE_LIB/bin:$ANACONDA_HOME/bin:$PATH

export LD_LIBRARY_PATH=$ANACONDA_HOME/lib:$CAFFE_LIB/lib:$CAFFE_LIB/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/limin/Repo/BMXNet/build
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1  

set -e

python /home/limin/Repo/BMXNet/smd_hpi/examples/bssd/evaluate.py \
	  --rec-path /home/limin/Repo/BMXNet/smd_hpi/examples/bssd/data/val.rec \
	  --prefix /home/limin/Repo/BMXNet/smd_hpi/examples/bssd/model/ssd_300/bssd_ \
	  --network vgg16_reduced --data-shape 300 --epoch 99 \
	  --gpus 0 --batch-size 32 \

#> /home/limin/Repo/BMXNet/smd_hpi/examples/bssd/val_bssd300.log  2>&1 | less \


