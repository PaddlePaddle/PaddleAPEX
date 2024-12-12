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
        if not cfg.disable_dump_func_state:
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
                api_recorder.update_real_data(args, kwargs)
                # print(self.op_name_)
                output = getattr(HookOp, "wrap_" + str(self.op_name_))(*args, **kwargs)
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
        else:
            output = getattr(HookOp, "wrap_" + str(self.op_name_))(*args, **kwargs)
        return output

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)


def temp_init(self, *inputs, **kwargs):
    # print("============init==================")
    self.cls_all_name_ = self.__class__.__name__
    # print("self.__class__.__name__ = ", self.cls_all_name_)
    cfg.prefix_op_name_ = self.cls_all_name_ + "*"
    cfg.disable_dump_func_state = True
    super(self.__class__, self).__init__(*inputs, **kwargs)

def temp_forward(self, *inputs, **kwargs):
    # print("============forward==================")
    if self.cls_all_name_ not in cfg.Op_count:
        cfg.Op_count[self.cls_all_name_] = 1
        cfg.prefix_op_name_ += "0"
    else:
        cfg.Op_count[self.cls_all_name_] += 1
        cfg.prefix_op_name_ += str(cfg.Op_count[self.cls_all_name_] - 1)
    if cfg.dump_state:
        api_recorder = API(cfg.dump_mode)
        rank = dist.get_rank()
        api_recorder.update_APIInfo(cfg.prefix_op_name_, rank)

        extra_param_str = api_recorder.get_extra_param()
        extra_param = {param: getattr(self, param.split('.')[1]) for param in extra_param_str}

        # print("extra_param: \n", extra_param)
        api_recorder.update_real_data(inputs, kwargs, extra_param, "class")
        # Call the parent class function
        # print(self.cls_all_name_ + '.forword')
        output = super(self.__class__, self).forward(*inputs, **kwargs)
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
            print(self.cls_all_name_+ '.forword', " register hook failed. Due to :", e)
            api_recorder.record_dout(None)
    else:
        output = super(self.__class__, self).forward(*inputs, **kwargs)

    cfg.disable_dump_func_state = False
    # print(output)
    return output

