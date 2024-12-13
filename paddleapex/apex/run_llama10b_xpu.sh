#!/bin/bash
task_name_or_path="llama-10b"
#export XPUAPI_DEBUG=0x1
#export XPURT_DISPATCH_MODE=PROFILING
export XPU_FORCE_USERMODE_LAUNCH=1
export PYTHONPATH=$PYTHONPATH:/workspace/APEX/PaddleAPEX:/workspace/APEX/PaddleNLP

runtime_location=/workspace/so-runtime
bkcl_location=/workspace/so-bkcl
export LD_LIBRARY_PATH=${bkcl_location}/:${runtime_location}/:$LD_LIBRARY_PATH

export XBLAS_FC_HBM_VERSION=40

# PaddlePaddle
export FLAGS_use_stride_kernel="0"
# export XPU_CDNN_CLUSTER_PARALLEL=1
# export XPU_CDNN_CLUSTER_PARALLEL_STREAM_NUMBER=2
export XPU_PADDLE_L3_SIZE0=1024
export XPU_PADDLE_L3_SIZE1=1024

export XPUAPI_DEBUG=0x1

# BKCL
# export BKCL_DEBUG=1
# Multi-computer RDMA
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=0
#export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
#export BKCL_SOCKET_IFNAME=eth0
export BKCL_FORCE_L3_RDMA=0
export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_RING_HOSTID_USE_RANK=1

echo "bkcl version:"
strings ${bkcl_location}/libbkcl.so | grep COM
master_ip=$POD_0_IP
nnodes=$PADDLE_TRAINERS_NUM
echo "master ip:"
echo $master_ip

export CUDA_DEVICE_MAX_CONNECTIONS=8

timestamp=$(date +%Y%m%d%H%M%S)
echo $timestamp

PaddleNLP_DIR=$(pwd)
echo "PaddleNLP_DIR: "$PaddleNLP_DIR

export USING_LAYERNORM=1
export USING_GQA_NEOX=1
export XPU_PADDLE_FUSE_SHARDING_BUFFER=1

export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_RING_HOSTID_USE_RANK=1

export USING_LOGITS_PRINT=1
export LOGITS_PRINT_INTERVAL=1
export XPU_PADDLE_FC_LOCAL_INT16=1


python run_paddle.py -json /workspace/APEX/llama10b/dump_info/rank0_step0/forward_rank0_all.json -backend xpu -out /workspace/APEX/llama10b/ -mode acc
