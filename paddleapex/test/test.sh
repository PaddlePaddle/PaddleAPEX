#！/bin/bash

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <backend>"
    exit 1
fi

BACKEND=$1
echo "The provided backend is: $BACKEND"
export PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

python test_demo.py
cp -r ../apex ./
python ./apex/run_paddle.py -json ./dump_info/rank0_step0/forward_rank0.json -out ./test_pipeline_force -backend $BACKEND \
 -mode all -dtype FP32,BF16

python ./apex/run_paddle.py -json ./dump_info/rank0_step0/forward_rank0.json -out ./test_pipeline_origin_data -backend $BACKEND \
 -mode acc 

python ./apex/acc_direct_cmp.py -bench ./test_pipeline_force/FP32 -device ./test_pipeline_force/BF16 -o ./direct_cmp_test_BF16

python ./apex/prof_cmp.py -bench ./test_pipeline_force/BF16 -device ./test_pipeline_force/BF16 -o ./prof_cmp_test_BF16
python ./apex/mem_cmp.py -bench ./test_pipeline_force/FP32 -device ./test_pipeline_force/FP32 -o ./prof_cmp_test_FP32

python ./apex/acc_direct_cmp.py -bench ./test_pipeline_force/BF16 -device ./test_pipeline_force/BF16 -o ./direct_cmp_test_BF16

