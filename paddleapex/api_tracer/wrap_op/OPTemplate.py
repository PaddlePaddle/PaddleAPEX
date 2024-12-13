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
from inspect import signature
import os
import pickle


class HookOp:
    pass


cfg = config.cfg


def hijack_init(self, *args, **kwargs):
    print("args", args)
    print("kwargs", kwargs)
    self.__init__(*args, **kwargs)


# 获取初始化参数的方法
def get_init_params(instance):
    sig = signature(instance.__init__)
    # 获取参数名称及默认值
    bound_args = sig.bind_partial()
    bound_args.apply_defaults()
    
    # 提取参数值
    init_params = {}
    for param in sig.parameters.values():
        if param.name != 'self':
            init_params[param.name] = getattr(instance, param.name, param.default)
    
    return init_params


def save_init_params_and_weight(init_params, state_dict, name, rank):
    data_route = cfg.dump_root_path
    directory = os.path.join(data_route, f"rank{rank}_step{cfg.global_step}")
    file_path = os.path.join(directory, f"{name}.init_params")
    with open(file_path, 'wb') as f:
        pickle.dump(init_params, f)
    # paddle.save(init_params, file_path)
    paddle.save(state_dict, os.path.join(directory, f"{name}.state_dict"))


def hijack_call(self, *args, **kwargs):
    cls = self.__class__
    init_params = get_init_params(self)
    # print("init_params", init_params)
    # print("hijack_call", self.__class__.__name__)
    cfg.prefix_op_name_ = self.prefix_op_name_ + "*"
    if self.__class__.__name__ not in cfg.Op_count:
        cfg.Op_count[self.__class__.__name__] = 1
        cfg.prefix_op_name_ += "0"
    else:
        cfg.Op_count[self.__class__.__name__] += 1
        cfg.prefix_op_name_ += str(cfg.Op_count[self.__class__.__name__] - 1)
    if cfg.dump_state:
        api_recorder = API(cfg.dump_mode)
        rank = dist.get_rank()
        api_recorder.update_APIInfo(cfg.prefix_op_name_, rank)
        api_recorder.update_real_data(args, kwargs)
        save_init_params_and_weight(init_params, self.state_dict(), cfg.prefix_op_name_, rank)
        output = self.forward(*args, **kwargs)
        # api_recorder.update_output(output)
        # print("api_info_struct !!!!!!", api_recorder.api_info_struct)
        # print(output)
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
            print(self.__class__.__name__, " register hook failed. Due to :", e)
            api_recorder.record_dout(None)
    else:
        output = self.forward(*args, **kwargs)
    return output



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
            api_recorder.update_real_data(args, kwargs)
            output = getattr(HookOp, "wrap_" + str(self.op_name_))(*args, **kwargs)
            # api_recorder.update_output(output)
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
