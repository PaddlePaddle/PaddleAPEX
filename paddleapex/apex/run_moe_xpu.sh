ask_name_or_path="llama-moe"
export XPU_FORCE_USERMODE_LAUNCH=1
export PYTHONPATH=$PYTHONPATH:/ssd3/zhouxiangquan/PaddleAPEX:/ssd3/zhouxiangquan/PaddleNLP

export XBLAS_FC_HBM_VERSION=40

# PaddlePaddle
export FLAGS_use_stride_kernel="0"
export XPU_CDNN_CLUSTER_PARALLEL=1
export XPU_CDNN_CLUSTER_PARALLEL_STREAM_NUMBER=2
export XPU_PADDLE_L3_SIZE0=1024
export XPU_PADDLE_L3_SIZE1=1024

# BKCL
export BKCL_TREE_THRESHOLD=0

# PaddlePaddle
export FLAGS_use_stride_kernel="0"

export CUDA_DEVICE_MAX_CONNECTIONS=8
#export XPUAPI_DEBUG=0x1
timestamp=$(date +%Y%m%d%H%M%S)
echo $timestamp

PaddleNLP_DIR=$(pwd)

export XPU_CHECKPOINT_ALLGATHER_OFFLOAD=1

export USING_LAYERNORM=1
export USING_GQA_NEOX=1
export USING_LOGITS_PRINT=1
export LOGITS_PRINT_INTERVAL=1
export XPU_PADDLE_FC_LOCAL_INT16=1
# --resume_from_checkpoint "/workspace/mnt/moe_workspace/llama-moe-gpu-checkpoint-2" \


#python -u  -m paddle.distributed.launch  --xpus "0,1,2,3,4,5,6,7" acc_direct_paddle.py --bench /ssd3/zhouxiangquan/moe/GPU/ --device /ssd3/zhouxiangquan/moe/result/ -o /ssd3/zhouxiangquan/moe/

python -u  -m paddle.distributed.launch  --xpus "0,1,2,3,4,5,6,7" run_paddle.py \
           -json \
           "/ssd3/zhouxiangquan/moe/dump_info/rank0_step0/forward_rank0_all.json /ssd3/zhouxiangquan/moe/dump_info/rank1_step0/forward_rank1_all.json /ssd3/zhouxiangquan/moe/dump_info/rank2_step0/forward_rank2_all.json /ssd3/zhouxiangquan/moe/dump_info/rank3_step0/forward_rank3_all.json /ssd3/zhouxiangquan/moe/dump_info/rank4_step0/forward_rank4_all.json /ssd3/zhouxiangquan/moe/dump_info/rank5_step0/forward_rank5_all.json /ssd3/zhouxiangquan/moe/dump_info/rank6_step0/forward_rank6_all.json /ssd3/zhouxiangquan/moe/dump_info/rank7_step0/forward_rank7_all.json" \
           -backend xpu \
           -real \
           "/ssd3/zhouxiangquan/moe/dump_info/rank0_step0/ /ssd3/zhouxiangquan/moe/dump_info/rank1_step0/ /ssd3/zhouxiangquan/moe/dump_info/rank2_step0/ /ssd3/zhouxiangquan/moe/dump_info/rank3_step0/ /ssd3/zhouxiangquan/moe/dump_info/rank4_step0/ /ssd3/zhouxiangquan/moe/dump_info/rank5_step0/ /ssd3/zhouxiangquan/moe/dump_info/rank6_step0/ /ssd3/zhouxiangquan/moe/dump_info/rank7_step0/" \
           -out /ssd3/zhouxiangquan/moe/result/ -mode pro -class 1 -dist 1

