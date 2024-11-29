#!/bin/bash
task_name_or_path="llama-20b"
export XPUAPI_DEBUG=0x1
#export XPURT_DISPATCH_MODE=PROFILING
export XPU_FORCE_USERMODE_LAUNCH=1
export PYTHONPATH=$PYTHONPATH:/workspace/APEX/PaddleAPEX:/workspace/APEX/PaddleNLP

export XBLAS_FC_HBM_VERSION=40

# PaddlePaddle
export FLAGS_use_stride_kernel="0"
#export XPU_CDNN_CLUSTER_PARALLEL=1
#export XPU_CDNN_CLUSTER_PARALLEL_STREAM_NUMBER=2
export XPU_PADDLE_L3_SIZE0=1024
export XPU_PADDLE_L3_SIZE1=1024
export XPU_PADDLE_FC_LOCAL_INT16=1

# BKCL
# export BKCL_DEBUG=1
# Multi-computer RDMA
#export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=0
export BKCL_TREE_THRESHOLD=0
#export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
#export BKCL_SOCKET_IFNAME=eth0
export BKCL_FORCE_L3_RDMA=0

export CUDA_DEVICE_MAX_CONNECTIONS=8
export CUDA_DEVICE_ORDER=OAM_ID

timestamp=$(date +%Y%m%d%H%M%S)
echo $timestamp

PaddleNLP_DIR=$(pwd)

export USING_LAYERNORM=1
export USING_GQA_NEOX=1
export USING_LOGITS_PRINT=1
export LOGITS_PRINT_INTERVAL=1

python -u  -m paddle.distributed.launch  --xpus "0,1,2,3,4,5,6,7" run_distributed.py -json ./ -backend xpu -out /workspace/APEX/llama20b/distributed/ -mode pro
# python -u  -m paddle.distributed.launch  --xpus "0,1,2,3,4,5,6,7" run_distributed.py -json ./ -backend xpu -out /workspace/APEX/llama20b/distributed/ -mode acc
# python run_paddle.py -json /workspace/APEX/llama20b/dump_info/rank0_step0/forward_rank0_all.json -backend xpu -out /workspace/APEX/llama20b/ -mode acc
# python run_paddle.py -real /workspace/APEX/scaled_dot_product_attention/dump_info/rank0_step0/ -json /workspace/APEX/scaled_dot_product_attention/dump_info/rank0_step5/forward_rank0_all.json -backend xpu -out /workspace/APEX/scaled_dot_product_attention/ -mode acc
