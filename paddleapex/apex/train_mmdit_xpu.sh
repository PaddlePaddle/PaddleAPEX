#!/bin/bash

mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
node_rank=$((mpi_rank+offset))
mpi_node=${OMPI_COMM_WORLD_SIZE:-1}
echo "MPI status:${mpi_rank}/${mpi_node}"
nnode_train=${nnode_set:-${mpi_node}}
master_train=${master:-localhost}

echo "Distributed Training ${node_rank}/${nnode_train} master=${master_train}"
set -x

nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

#source ./script/utils.sh
for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINERS_NUM
unset PADDLE_TRAINER_ID
unset PADDLE_WORKERS_IP_PORT_LIST
unset PADDLE_TRAINERS
unset PADDLE_NUM_GRADIENT_SERVERS

export XPU_FORCE_USERMODE_LAUNCH=1

runtime_location=/workspace/so-runtime
bkcl_location=/workspace/so-bkcl
fast_paddle_location=/workspace/so-fast_paddle
export LD_LIBRARY_PATH=${bkcl_location}/:${runtime_location}/:${fast_paddle_location}/:$LD_LIBRARY_PATH

export XBLAS_FC_HBM_VERSION=40

# PaddlePaddle
export FLAGS_use_stride_kernel="0"
export XPU_CDNN_CLUSTER_PARALLEL=1
export XPU_CDNN_CLUSTER_PARALLEL_STREAM_NUMBER=2
export XPU_PADDLE_L3_SIZE0=1024
export XPU_PADDLE_L3_SIZE1=1024
#export XPUAPI_DEFAULT_SIZE0=1502653248
#export XPUAPI_DEFAULT_SIZE1=380265324
export XPU_PADDLE_FUSE_SHARDING_BUFFER=1

# BKCL
# Multi-computer RDMA
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=0
export BKCL_FORCE_L3_RDMA=0
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
export BKCL_SOCKET_IFNAME=eth0
echo "bkcl version:"
strings ${bkcl_location}/libbkcl.so | grep COM

export CUDA_DEVICE_MAX_CONNECTIONS=8
export BKCL_FLAT_RING=1

master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
port=36677

export PYTHONPATH=/workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/PaddleAPEX:/workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/PaddleMIX/ppdiffusers:/workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/PaddleMIX/PaddleNLP:$PYTHONPATH

tp2pp4=${tp2pp4:-"False"}
if [ ${tp2pp4} == "True" ];then
    unset BKCL_RDMA_NICS
    unset CUDA_DEVICE_ORDER
    unset XPULINK_VISIBLE_DEVICES

    export CUDA_DEVICE_ORDER=OAM_ID
    export XPULINK_VISIBLE_DEVICES=2,3,0,1,4,5,6,7
    export BKCL_RDMA_NICS=eth2,eth2,eth1,eth1,eth3,eth3,eth4,eth4
fi

export BKCL_USE_AR=1
# export BKCL_RING_OPT=1
export BKCL_RING_HOSTID_USE_RANK=1

# accuracy improve: matmul with  fp32 input will use fp32 to calc instead of using int16
# export XPU_PADDLE_FC_LOCAL_INT16=1
export XPU_AUTO_BF16_TF32=1
export XPU_PADDLE_FC_TF32=1

# memory improve
# export XPU_INPLACE_SHARING_BF16_FP16_CACHE=1

export CUDA_DISABLE_PRINTF=1
export BCCL_TRACE_HANG_ENABLE=1
export BCCL_HANG_DETECT_INTERVAL=5
export BCCL_UNIX_SOCKET_PATH=/var/run
export BCCL_ERROR_FILE=/root/paddlejob/workspace/log/err.%h.%p.log

if [[ $rank -ge $nnodes ]]; then
    exit 0
fi

timestamp=$(date +%Y%m%d%H%M%S)
echo $timestamp

# open it when debug
export XPUAPI_DEBUG=0x1
#export XPURT_DISPATCH_MODE="PROFILING"
#export GLOG_v=10

python -m paddle.distributed.launch --xpus "0,1,2,3,4,5,6,7" run_paddle.py -json \
    "/workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank0_step5/forward_rank0_all.json /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank1_step5/forward_rank1_all.json /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank2_step5/forward_rank2_all.json /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank3_step5/forward_rank3_all.json /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank4_step5/forward_rank4_all.json /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank5_step5/forward_rank5_all.json /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank6_step5/forward_rank6_all.json /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank7_step5/forward_rank7_all.json" \
    -backend xpu \
    -real \
    "/workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank0_step5/ /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank1_step5/ /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank2_step5/ /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank3_step5/ /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank4_step5/ /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank5_step5/ /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank6_step5/ /workspace/ZHOU/baidu/personal-code/dit_t2iv_to_kunlun/dump_info/rank7_step5/" \
    -out result_xpu/ -mode acc -class 1 -class_type float16 -dist 1

