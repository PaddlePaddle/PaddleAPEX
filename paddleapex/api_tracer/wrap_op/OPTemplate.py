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
import paddle.distributed as dist
from .. import config
from ..api_info import API


class HookOp:
    pass


cfg = config.cfg


class OPTemplate:
    def __init__(self, op_name):
        self.op_name_ = op_name
        cfg.prefix_op_name_ = self.op_name_ + "*"

    def forward(self, *args, **kwargs):
        if self.op_name_ not in cfg.Op_count:
            cfg.Op_count[self.op_name_] = 1
            cfg.prefix_op_name_ += "0"
        else:
            cfg.Op_count[self.op_name_] += 1
            cfg.prefix_op_name_ += str(cfg.Op_count[self.op_name_] - 1)
        if cfg.dump_state:
            api_recorder = API(cfg.dump_mode)
            rank = dist.get_rank()
            api_recorder.update_APIInfo(cfg.prefix_op_name_, rank)
            output = getattr(HookOp, "wrap_" + str(self.op_name_))(*args, **kwargs)
            api_recorder.update_real_data(args, kwargs)
        else:
            output = getattr(HookOp, "wrap_" + str(self.op_name_))(*args, **kwargs)
        return output

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)
