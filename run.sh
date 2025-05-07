export XPUAPI_DEBUG=0x1
export CUDA_VISIBLE_DEVICES=7
XPU_AUTO_BF16_TF32=1 XPU_PADDLE_FC_TF32=1 python test_fc.py
python test_fa.py
cd paddleapex/apex/
python run_paddle.py -json ../../dump_info/rank0_step0/forward_rank0_all.json -real ../../dump_info/rank0_step0/ -out XPU -backend XPU -mode acc
cd ../../../linux-bcecmd-0.3.0
./bcecmd --conf-path ./ bos sync ../PaddleAPEX bos:/baidu-kunlun-customer/NEW_FA_0312/
