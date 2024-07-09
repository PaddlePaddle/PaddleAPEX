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
import paddle
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
            try:
                if isinstance(output, paddle.Tensor):
                    if not output.stop_gradient:
                        output.register_hook(api_recorder.record_dout)
                        api_recorder.output_num = 1
                    else:
                        api_recorder.record_dout(None)
                if isinstance(output, (list, tuple)):
                    need_record = False
                    for item in output:
                        if isinstance(item, paddle.Tensor) and not item.stop_gradient:
                            api_recorder.output_num += 1
                            need_record = True
                            item.register_hook(api_recorder.record_dout)
                    if not need_record:
                        api_recorder.record_dout(None)
            except Exception as e:
                print(self.op_name_, " register hook failed. Due to :", e)
                api_recorder.record_dout(None)
        else:
            output = getattr(HookOp, "wrap_" + str(self.op_name_))(*args, **kwargs)
        return output

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)
