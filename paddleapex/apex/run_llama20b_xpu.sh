#!/bin/bash
task_name_or_path="llama-20b"
#export XPUAPI_DEBUG=0x1
#export XPURT_DISPATCH_MODE=PROFILING
export XPU_FORCE_USERMODE_LAUNCH=1
export PYTHONPATH=$PYTHONPATH:/ssd3/zhouxiangquan/PaddleAPEX:/ssd3/zhouxiangquan/PaddleNLP

export XBLAS_FC_HBM_VERSION=40

# PaddlePaddle
export FLAGS_use_stride_kernel="0"
export XPU_CDNN_CLUSTER_PARALLEL=1
export XPU_CDNN_CLUSTER_PARALLEL_STREAM_NUMBER=2
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
export XPU_CHECKPOINT_ALLGATHER_OFFLOAD=1
export XPU_AUTO_BF16_TF32_RADIO=1
export XPU_AUTO_BF16_TF32=1  # 开启TF32/BF16自动切换
export XPU_AUTO_BF16_TF32_DEBUG=1   # 开启日志打印

timestamp=$(date +%Y%m%d%H%M%S)
echo $timestamp

PaddleNLP_DIR=$(pwd)

export USING_LAYERNORM=1
export USING_GQA_NEOX=1
export USING_LOGITS_PRINT=1
export LOGITS_PRINT_INTERVAL=1


#python -u  -m paddle.distributed.launch  --xpus "0,1,2,3,4,5,6,7" acc_direct_cmp_zxq.py --bench /ssd3/zhouxiangquan/llama20b/GPU/ --device /ssd3/zhouxiangquan/llama20b/result/rank_0/ -o /ssd3/zhouxiangquan/llama20b/
#python lot_t.py

#python run_paddle.py -json /ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/test.json -backend xpu -real /ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/ -out /ssd3/zhouxiangquan/llama20b/result/ -mode acc

python -u  -m paddle.distributed.launch  --xpus "0,1,2,3,4,5,6,7" run_paddle.py \
           -json \
           "/ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/forward_rank0_all.json /ssd3/zhouxiangquan/llama20b/dump_info/rank1_step0/forward_rank1_all.json /ssd3/zhouxiangquan/llama20b/dump_info/rank2_step0/forward_rank2_all.json /ssd3/zhouxiangquan/llama20b/dump_info/rank3_step0/forward_rank3_all.json /ssd3/zhouxiangquan/llama20b/dump_info/rank4_step0/forward_rank4_all.json /ssd3/zhouxiangquan/llama20b/dump_info/rank5_step0/forward_rank5_all.json /ssd3/zhouxiangquan/llama20b/dump_info/rank6_step0/forward_rank6_all.json /ssd3/zhouxiangquan/llama20b/dump_info/rank7_step0/forward_rank7_all.json" \
           -backend xpu \
           -real \
           "/ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank1_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank2_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank3_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank4_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank5_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank6_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank7_step0/" \
           -out /ssd3/zhouxiangquan/llama20b/result/ -mode pro -class 1 -dist 1

#
#python -u  -m paddle.distributed.launch  --xpus "0,1,2,3,4,5,6,7" run_distributed.py \
#           -json \
#           "/ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/distributed.json /ssd3/zhouxiangquan/llama20b/dump_info/rank1_step0/distributed.json /ssd3/zhouxiangquan/llama20b/dump_info/rank2_step0/distributed.json /ssd3/zhouxiangquan/llama20b/dump_info/rank3_step0/distributed.json /ssd3/zhouxiangquan/llama20b/dump_info/rank4_step0/distributed.json /ssd3/zhouxiangquan/llama20b/dump_info/rank5_step0/distributed.json /ssd3/zhouxiangquan/llama20b/dump_info/rank6_step0/distributed.json /ssd3/zhouxiangquan/llama20b/dump_info/rank7_step0/distributed.json" \
#           -backend xpu \
#           -real \
#           "/ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank1_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank2_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank3_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank4_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank5_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank6_step0/ /ssd3/zhouxiangquan/llama20b/dump_info/rank7_step0/" \
#           -out /ssd3/zhouxiangquan/llama20b/result/ -mode acc
#
#
#python -u  -m paddle.distributed.launch  --xpus "0,1,2,3,4,5,6,7" run_without_distributed.py \
#           -json \
#           "/ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/common.json" \
#           -backend xpu \
#           -real \
#           "/ssd3/zhouxiangquan/llama20b/dump_info/rank0_step0/" \
#           -out /ssd3/zhouxiangquan/llama20b/result/rank_0/ -mode acc
#

