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
import numpy as np
from .Dump import dump_util


def transfer_types(data, dtype):
    if "INT" in dtype or "BOOL" in dtype:
        return int(data)
    else:
        return float(data)


def get_type_name(name):
    left = name.index("'")
    right = name.rindex("'")
    return name[left + 1 : right]


def get_tensor_extremum(data, operator):
    def handle_tensor_extremum_nan_inf(data_clone, operator):
        data_nan = np.isnan(data_clone)
        if int(np.sum(data_nan)) == data_clone.numel():
            return float("nan")
        finite_mask = np.isfinite(data_clone)
        if int(np.sum(finite_mask)) > 0:
            finite_values = data_clone[finite_mask]
            return finite_values if operator == "max" else finite_values
        else:
            data_no_nan = data_clone[~data_nan]
            return np.max(data_no_nan) if operator == "max" else np.min(data_no_nan)

    if data.dtype is paddle.bool:
        if data.numel() == 0:
            return False, False
        if operator == "max":
            return True in data, True in data
        elif operator == "min":
            return False not in data, False not in data
    data_clone = data.clone().detach().numpy()
    if operator == "max":
        max_result = np.max(data_clone).item()
        if np.isinf(max_result) or np.isnan(max_result):
            return handle_tensor_extremum_nan_inf(data_clone, operator), max_result
        else:
            return max_result, max_result
    else:
        min_result = np.min(data_clone).item()
        if np.isinf(min_result) or np.isnan(min_result):
            return handle_tensor_extremum_nan_inf(data_clone, operator), min_result
        else:
            return min_result, min_result


class API:
    def __init__(self, mode):
        self.op_name = ""
        self.rank = ""
        self.args = ""
        self.kwargs = ""
        self.mode = mode
        self.args_num = 0

    """
        Adjust data format.
        Transfer opinfo_dict to dump utils
    """

    def reformat(self):
        args_info_list = self.analyze_element(self.args)
        kwargs_info_dict = self.analyze_element(self.kwargs)
        self.api_info_struct = {
            self.op_name: {"args": args_info_list, "kwargs": kwargs_info_dict}
        }
        dump_util.update_api_dict(self.api_info_struct, self.rank)

    def update_APIInfo(self, op_name, rank):
        print("dump api: ", op_name)
        self.op_name = op_name
        self.rank = rank

    def update_real_data(self, output, inputs, kwargs):
        self.args = inputs
        self.kwargs = kwargs
        self.output = output
        self.reformat()

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

        if isinstance(element, paddle.Tensor):
            return self._analyze_tensor(element)

        if element is None or isinstance(element, (bool, int, float, str, slice)):
            return self._analyze_builtin(element)

        msg = f"In op:{self.op_name}, its args type {type(element)} is unsupported at analyze_element"
        print(msg)

    def _analyze_tensor(self, arg):
        single_arg = {}
        single_arg.update({"type": "paddle.Tensor"})
        single_arg.update({"dtype": str(arg.dtype.name)})
        single_arg.update({"shape": arg.shape})
        if arg.dtype.name == "BF16":
            arg = paddle.cast(arg, "float32")
        max_handle, max_origin = get_tensor_extremum(arg, "max")
        single_arg.update({"Max": transfer_types(max_handle, str(arg.dtype.name))})
        single_arg.update(
            {"Max_origin": transfer_types(max_origin, str(arg.dtype.name))}
        )
        min_handle, min_origin = get_tensor_extremum(arg, "min")
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
            np.bytes_: bytes,
            np.unicode_: str,
        }
        for numpy_type, builtin_type in type_mapping.items():
            if isinstance(arg, numpy_type):
                return builtin_type(arg), get_type_name(str(type(arg)))
        return arg, ""
