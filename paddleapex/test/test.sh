
export PYTHONPATH=./:$PYTHONPATH
python test_demo.py
python ./apex/run_paddle.py -json ./dump_info/rank0_step0/forward_rank0.json -out ./test_pipeline_force_FP32 -backend gpu \
 -mode all -dtype FP32
python ./apex/run_paddle.py -json ./dump_info/rank0_step0/forward_rank0.json -out ./test_pipeline_force_BF16 -backend gpu \
 -mode all -dtype BF16


python ./apex/run_paddle.py -json ./dump_info/rank0_step0/forward_rank0.json -out ./test_pipeline_origin_data -backend gpu \
 -mode acc 

python ./apex/acc_direct_cmp.py -bench ./test_pipeline_force_FP32 -device ./test_pipeline_force_BF16 -o ./direct_cmp_test_BF16

python ./apex/prof_cmp.py -bench ./test_pipeline_force_FP32 -device ./test_pipeline_force_BF16 -o ./prof_cmp_test_BF16
python ./apex/mem_cmp.py -bench ./test_pipeline_force_FP32 -device ./test_pipeline_force_BF16 -o ./prof_cmp_test_BF16




