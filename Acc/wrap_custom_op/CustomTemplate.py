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
from ..api_info import API
from .. import config
import paddle.nn as nn
import paddle.distributed as dist
cfg = config.cfg


class HookCustomOp:
    pass




class FunctionalOPTemplate():
    def __init__(self, op_name):
        self.op_name_ = op_name
        cfg.prefix_functional_op_name_ = "Custom*" + str(op_name) + "*"
        
    def forward(self, *inputs, **kwargs):

        if self.op_name_ not in cfg.Custom_op_count:
            cfg.Custom_op_count[self.op_name_] = 1
            cfg.prefix_custom_op_name_ += "0"
        else:
            cfg.Custom_op_count[self.op_name_] += 1
            cfg.prefix_functional_op_name_ += str(
                cfg.Custom_op_count[self.op_name_] - 1
            )

        if cfg.dump_state:
            api_recorder = API(cfg.dump_mode)
            rank = dist.get_rank()
            api_recorder.update_APIInfo(cfg.prefix_custom_op_name_, rank)
            output = getattr(HookCustomOp, "wrap_" + str(self.op_name_))(
                *inputs, **kwargs
            )
            api_recorder.update_real_data(output, inputs, kwargs)
        else:
            output = getattr(HookCustomOp, "wrap_" + str(self.op_name_))(
                *inputs, **kwargs
            )
        return output

    def __call__(self, *inputs, **kwargs):
        print("dump api: ", self.prefix_custom_op_name_)
        return self.forward(*inputs, **kwargs)
