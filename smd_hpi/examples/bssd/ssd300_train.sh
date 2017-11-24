#!/bin/sh

#PBS -N voc0712_bssd300
#PBS -l nodes=gpu02
#PBS -l walltime=24:00:00
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

python /home/limin/Repo/BMXNet/smd_hpi/examples/bssd/train.py --gpus 0 --batch-size 32 \
	  --train-path /home/limin/Repo/BMXNet/smd_hpi/examples/bssd/data/train.rec \
	  --val-path /home/limin/Repo/BMXNet/smd_hpi/examples/bssd/data/val.rec   \
      --pretrained /home/limin/Repo/BMXNet/smd_hpi/examples/bssd/model/ssd_300/bssd_vgg16_reduced_300 \
	  --prefix /home/limin/Repo/BMXNet/smd_hpi/examples/bssd/model/ssd_300/bssd \
	  --network vgg16_reduced --lr 0.0002 \
	  --data-shape 300 --epoch 99 --resume 99 \
       > /home/limin/Repo/BMXNet/smd_hpi/examples/bssd/train_ssd300_3.log 2>&1

