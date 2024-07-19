"""
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
"""
import paddle
import os
import numpy
import math
import random
import numpy as np
from .utils import (
    check_object_type,
    CompareException,
    check_file_or_directory_path,
)
from .logger import (
    print_error_log,
    print_warn_log,
)

seed = 1234
TENSOR_DATA_LIST_PADDLE = ["paddle.Tensor", "paddle.create_parameter"]
PADDLE_TYPE = ["paddle.CPUPlace", "paddle.Tensor.dtype"]
FLOAT_TYPE_PADDLE = [
    "FP16",
    "FP32",
    "FP64",
    "BF16",
    "paddle.float",
    "paddle.float64",
    "paddle.double",
    "paddle.float16",
    "paddle.half",
    "paddle.bfloat16",
]
REAL_TYPE_PADDLE = {
    "FP64": "paddle.float64",
    "FP32": "paddle.float32",
    "BF16": "paddle.bfloat16",
    "FP16": "paddle.float16",
    "BOOL": "paddle.bool",
    "UINT8": "paddle.uint8",
    "INT16": "paddle.int16",
    "INT32": "paddle.int32",
    "INT64": "paddle.int64",
}
NUMPY_TYPE = [
    "numpy.int8",
    "numpy.int16",
    "numpy.int32",
    "numpy.int64",
    "numpy.uint8",
    "numpy.uint16",
    "numpy.uint32",
    "numpy.uint64",
    "numpy.float16",
    "numpy.float32",
    "numpy.float64",
    "numpy.float128",
    "numpy.complex64",
    "numpy.complex128",
    "numpy.complex256",
    "numpy.bool_",
    "numpy.string_",
    "numpy.bytes_",
    "numpy.unicode_",
]


def gen_data(info, real_data_path=None):
    check_object_type(info, dict)
    data_type = info.get("type")
    rel_data_path = info.get("real_data_path")
    need_grad = False
    if data_type in TENSOR_DATA_LIST_PADDLE:
        stop_gradient = info.get("stop_gradient")
        if real_data_path:
            data_pth = os.path.join(real_data_path, rel_data_path)
            real_data_path = os.path.abspath(data_pth)
            if os.path.exists(real_data_path):
                data = gen_real_tensor(real_data_path, stop_gradient)
        else:
            data = gen_random_tensor(info, stop_gradient)
        data.stop_gradient = stop_gradient
        need_grad = not stop_gradient
    elif data_type.startswith("numpy"):
        if data_type not in NUMPY_TYPE:
            raise Exception("{} is not supported now".format(data_type))
        data = info.get("value")
        try:
            data = eval(data_type)(data)
        except Exception as err:
            print_error_log("Failed to convert the type to numpy: %s" % str(err))
    else:
        data = info.get("value")
        if info.get("type") == "slice":
            data = slice(*data)
    return data, need_grad


def gen_real_tensor(data_path, stop_gradient):
    data_path = os.path.realpath(data_path)
    check_file_or_directory_path(data_path)
    if not data_path.endswith(".pt") and not data_path.endswith(".npy"):
        error_info = f"The file: {data_path} is not a pt or numpy file."
        raise CompareException(CompareException.INVALID_FILE_ERROR, error_info)
    if data_path.endswith(".pt"):
        data = paddle.load(data_path)
    else:
        data_np = numpy.load(data_path)
        data = paddle.to_tensor(data_np)
    data.stop_gradient = stop_gradient
    return data


def gen_random_tensor(info, stop_gradient):
    check_object_type(info, dict)
    low, high = info.get("Min"), info.get("Max")
    low_origin, high_origin = info.get("Min_origin"), info.get("Max_origin")
    low_info = [low, low_origin]
    high_info = [high, high_origin]
    data_dtype = info.get("dtype")
    shape = tuple(info.get("shape"))
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        error_info = (
            f"Data info Min: {low} , Max: {high}, info type must be int or float."
        )
        raise CompareException(CompareException.INVALID_PARAM_ERROR, error_info)
    if data_dtype == "paddle.bool" or data_dtype == "BOOL":
        data = gen_bool_tensor(shape)
    else:
        data = gen_common_tensor(low_info, high_info, shape, data_dtype)
    data.stop_gradient = stop_gradient
    return data


def generate_random_tensor(shape, min_value, max_value):
    tensor = np.random.randn(*shape)
    tensor_min = np.min(tensor)
    tensor_max = np.max(tensor)
    tensor_normalized = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-6)
    tensor_scaled = (max_value - min_value) * tensor_normalized + min_value
    return tensor_scaled


def gen_common_tensor(low_info, high_info, shape, data_dtype):
    low = low_info[0]
    high = high_info[0]
    if data_dtype in FLOAT_TYPE_PADDLE:
        if math.isnan(high) or math.isnan(low) or math.isinf(high) or math.isinf(low):
            tensor = generate_random_tensor(shape, 0, 1)
            tensor = paddle.to_tensor(
                tensor, dtype=eval(REAL_TYPE_PADDLE.get(data_dtype))
            )
            return tensor
        else:
            if len(shape) == 0:
                shape = [1]
            tensor = generate_random_tensor(shape, low, high).astype(numpy.float32)
            tensor = paddle.to_tensor(
                tensor, dtype=eval(REAL_TYPE_PADDLE.get(data_dtype))
            )
            return tensor
    elif (
        "int" in data_dtype
        or "long" in data_dtype
        or "INT" in data_dtype
        or "LONG" in data_dtype
    ):
        low, high = int(low), int(high)
        if low == high:
            tensor = paddle.full(shape, low)
        else:
            tensor = numpy.random.randint(low, high, shape)
            tensor = paddle.to_tensor(
                tensor, dtype=eval(REAL_TYPE_PADDLE.get(data_dtype))
            )
        return tensor
    else:
        print_error_log("Dtype is not supported: " + data_dtype)
        raise NotImplementedError()


def gen_bool_tensor(shape):
    tensor = paddle.to_tensor(numpy.random.randint(0, 2, shape))
    data = paddle.cast(tensor, paddle.bool)
    return data


def gen_args(args_info, real_data_path = None, need_grad=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    check_object_type(args_info, list)
    args_result = []
    for arg in args_info:
        if isinstance(arg, (list, tuple)):
            data, has_grad = gen_args(arg, real_data_path, need_grad)
            need_grad = need_grad or has_grad
        elif isinstance(arg, dict):
            data, has_grad = gen_data(arg, real_data_path)
            need_grad = need_grad or has_grad
        elif isinstance(arg, str):
            data = eval(arg)
        else:
            print_warn_log(f"Warning: {arg} is not supported")
            raise NotImplementedError()
        args_result.append(data)
    return args_result, need_grad


def gen_kwargs(api_info, real_data_path=None):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    check_object_type(api_info, dict)
    kwargs_params = api_info.get("kwargs")
    need_grad = False
    for key, value in kwargs_params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            kwargs_params[key], has_grad_tensor = gen_list_kwargs(value, real_data_path)
            need_grad = need_grad or has_grad_tensor
        elif isinstance(value, str):
            kwargs_params[key] = eval(value)
        elif value.get("type") in TENSOR_DATA_LIST_PADDLE or value.get(
            "type"
        ).startswith("numpy"):
            kwargs_params[key], has_grad_tensor = gen_data(value, real_data_path)
            need_grad = need_grad or has_grad_tensor
        elif value.get("type") in PADDLE_TYPE:
            gen_paddle_kwargs(kwargs_params, key, value)
        else:
            kwargs_params[key] = value.get("value")
    return kwargs_params, need_grad


def gen_paddle_kwargs(kwargs_params, key, value):
    if value.get("type") != "paddle.CPUPlace":
        kwargs_params[key] = eval(value.get("value"))


def gen_list_kwargs(kwargs_item_value, real_data_path = None):
    kwargs_item_result = []
    has_grad_tensor = False
    for item in kwargs_item_value:
        if item.get("type") in TENSOR_DATA_LIST_PADDLE:
            item_value, has_grad_tensor = gen_data(item, real_data_path)
        else:
            has_grad_tensor = False
            item_value = item.get("value")
        kwargs_item_result.append(item_value)
    return kwargs_item_result, has_grad_tensor


def gen_api_params(api_info, real_data_path = None):
    check_object_type(api_info, dict)
    kwargs_params, kwargs_need_grad = gen_kwargs(api_info, real_data_path)
    if api_info.get("args"):
        args_params, args_need_grad = gen_args(api_info.get("args"), real_data_path)
    else:
        args_need_grad = False
        args_params = []
    need_grad = kwargs_need_grad or args_need_grad
    return args_params, kwargs_params, need_grad


def rand_like(data, seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if isinstance(data, paddle.Tensor):
        if data.dtype.name in ["BF16", "FP16"]:
            random_normals = numpy.random.randn(*data.shape)
            x = paddle.to_tensor(random_normals, dtype=data.dtype)
            return x
        elif data.dtype.name in ["FP32", "FP64"]:
            random_normals = numpy.random.randn(*data.shape)
            x = paddle.to_tensor(random_normals, dtype=data.dtype)
            return x
        elif data.dtype.name in ["INT32", "INT64"]:
            rand_data = numpy.random.randint(-10, 10, size=data.shape).astype("int")
            rand_data = paddle.to_tensor(rand_data, dtype=data.dtype)
            return rand_data
        else:
            raise ValueError(f"Unsupported dtype:{data.dtype.name} in func: rand_like")
    elif isinstance(data, (list, tuple)):
        lst = [rand_like(item) for item in data]
        return lst
