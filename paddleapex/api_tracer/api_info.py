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

import paddle
import math
import numpy as np
from paddleapex.api_tracer.Dump import dump_util
from paddleapex.api_tracer.config import cfg
from paddleapex.api_tracer.hook_op import HookOp
Paddle_Type_Map = {
    "FP64": "paddle.float64",
    "FP32": "paddle.float32",
    "BF16": "paddle.bfloat16",
    "FP16": "paddle.float16",
    "BOOL": "paddle.bool",
    "UINT8": "paddle.uint8",
    "INT16": "paddle.int16",
    "INT32": "paddle.int32",
    "INT64": "paddle.int64",
    "FLOAT64": "paddle.float64",
    "FLOAT32": "paddle.float32",
    "FLOAT16": "paddle.float16",
    "BFLOAT16": "paddle.bfloat16",
}

Half_Precision_List = [
    "BF16",
    "FP16",
    "BFLOAT16",
    "FLOAT16",
]

def get_rounded_num(x, round_up=True):
    if abs(x) <= 1e-10:
        return 0
    
    abs_x = abs(x)
    log_x = math.log10(abs_x)
    round_log_x = math.floor(log_x) if round_up ^ (x > 0) else math.ceil(log_x)
    
    result = 10**round_log_x
    return result if x >= 0 else -result

def get_type_name(name):
    left = name.index("'")
    right = name.rindex("'")
    return name[left + 1 : right]


def transfer_types(data, dtype):
    if "INT" in dtype or "BOOL" in dtype:
        return int(data)
    else:
        return float(data)


def get_tensor_extremum(data):
    if data.dtype is paddle.bool:
        if data.numel() == 0:
            return False, False, False, False
        if operator == "max":
            result = True in data
            return result, result, result, result
        elif operator == "min":
            result = False not in data
            return result, result, result, result
    data_clone = data.clone().detach().numpy()

    max_result = np.max(data_clone).item()
    min_result = np.min(data_clone).item()
    if cfg.dump_unique:
        ori_max_ = max_result
        ori_min_ = min_result
        max_result = get_rounded_num(ori_max_, True)
        min_result = get_rounded_num(ori_min_, False) if ori_min_ != ori_max_ else max_result
    return max_result, max_result, min_result, min_result


class API:
    def __init__(self, mode):
        self.op_name = ""
        self.rank = ""
        self.mode = mode
        self.args_num = 0
        self.embedding_num = 0
        self.output_num = 0
        self.dout_list = []
        self.is_half_precision = False
        if cfg.profile_mode:
            self.tensor_analyzer_ = self.effi_analyze_tensor
        else:
            self.tensor_analyzer_ = self._analyze_tensor

    def update_APIInfo(self, op_name, rank):
        print("dump api: ", op_name)
        self.op_name = op_name
        self.rank = rank

    def update_real_data(self, inputs, kwargs):
        self.is_half_precision = False
        args_info_list = self.analyze_element(inputs)
        kwargs_info_dict = self.analyze_element(kwargs)
        self.api_info_struct = {
            self.op_name: {"args": args_info_list, "kwargs": kwargs_info_dict, "dout_list": ["Failed"]}
        }
        dump_util.update_api_dict(self.api_info_struct, self.rank, self.is_half_precision)

    def record_dout(self, grad_value):
        if grad_value is not None:
            dout = self.analyze_element(grad_value)
            self.dout_list.append(dout)
            self.output_num -= 1
            if self.output_num == 0:
                self.api_info_struct[self.op_name].update({"dout_list": self.dout_list})

    def analyze_element(self, element):
        if isinstance(element, (list, tuple)):
            out = []
            for item in element:
                out.append(self.analyze_element(item))
            return out

        if isinstance(element, dict):
            out_dict = {}
            for key, value in element.items():
                out_dict[key] = self.analyze_element(value)
            return out_dict

        converted_numpy, numpy_type = self._convert_numpy_to_builtin(element)
        if converted_numpy is not element:
            return self._analyze_numpy(converted_numpy, numpy_type)

        if isinstance(element, paddle.dtype):
            if element.name in Half_Precision_List:
                self.is_half_precision = True
            return Paddle_Type_Map[element.name]

        if isinstance(element, paddle.Tensor):
            if element.dtype.name in Half_Precision_List:
                self.is_half_precision = True
            return self.tensor_analyzer_(element)

        if element is None or isinstance(element, (bool, int, float, str, slice)):
            return self._analyze_builtin(element)

        msg = f"In op:{self.op_name}, its args type {type(element)} is unsupported at analyze_element"
        print(msg)

    def effi_analyze_tensor(self, arg):
        single_arg = {}
        single_arg.update({"type": "paddle.Tensor"})
        single_arg.update({"dtype": str(arg.dtype.name)})
        single_arg.update({"shape": arg.shape})
        try:
            with paddle.no_grad():
                fun_max = getattr(HookOp, "wrap_paddle.max")
                fun_min = getattr(HookOp, "wrap_paddle.min")
                max_ = fun_max(arg).item()
                min_ = fun_min(arg).item()
        except:
            max_ = 1
            min_ = 0
        if cfg.dump_unique and arg.dtype.name != "BOOL":
            ori_max_ = max_
            ori_min_ = min_
            max_ = get_rounded_num(ori_max_, True)
            min_ = get_rounded_num(ori_min_, False) if ori_min_ != ori_max_ else max_
        single_arg.update({"Max": max_})
        single_arg.update({"Max_origin": max_})
        single_arg.update({"Min": min_})
        single_arg.update({"Min_origin": min_})
        single_arg.update({"stop_gradient": arg.stop_gradient})
        if self.mode == "real_data":
            api_args = self.op_name + "." + str(self.args_num)
            pt_path = dump_util.dump_real_data(api_args, arg.detach().cpu(), self.rank)
            self.args_num += 1
            single_arg.update({"real_data_path": pt_path})
        return single_arg

    def _analyze_tensor(self, arg):
        single_arg = {}
        single_arg.update({"type": "paddle.Tensor"})
        single_arg.update({"dtype": str(arg.dtype.name)})
        single_arg.update({"shape": arg.shape})
        if arg.dtype.name == "BF16":
            arg = paddle.cast(arg, "float32")
        max_handle, max_origin, min_handle, min_origin = get_tensor_extremum(arg)
        single_arg.update({"Max": transfer_types(max_handle, str(arg.dtype.name))})
        single_arg.update(
            {"Max_origin": transfer_types(max_origin, str(arg.dtype.name))}
        )
        single_arg.update({"Min": transfer_types(min_handle, str(arg.dtype.name))})
        single_arg.update(
            {"Min_origin": transfer_types(min_origin, str(arg.dtype.name))}
        )
        single_arg.update({"stop_gradient": arg.stop_gradient})

        if self.mode == "real_data":
            api_args = self.op_name + "." + str(self.args_num)
            pt_path = dump_util.dump_real_data(api_args, arg.detach().cpu(), self.rank)
            self.args_num += 1
            single_arg.update({"real_data_path": pt_path})
        return single_arg

    def _analyze_builtin(self, arg):
        single_arg = {}
        self.args_num += 1
        if isinstance(arg, slice):
            single_arg.update({"type": "slice"})
            single_arg.update({"value": [arg.start, arg.stop, arg.step]})
        else:
            single_arg.update({"type": get_type_name(str(type(arg)))})
            single_arg.update({"value": arg})
        return single_arg

    def _analyze_numpy(self, value, numpy_type):
        single_arg = {}
        self.args_num += 1
        single_arg.update({"type": numpy_type})
        single_arg.update({"value": value})
        return single_arg

    def _convert_numpy_to_builtin(self, arg):
        type_mapping = {
            np.integer: int,
            np.floating: float,
            np.bool_: bool,
            np.complexfloating: complex,
            np.str_: str,
            np.bytes_: bytes
            # np.unicode_: str,
        }
        for numpy_type, builtin_type in type_mapping.items():
            if isinstance(arg, numpy_type):
                return builtin_type(arg), get_type_name(str(type(arg)))
        return arg, ""
