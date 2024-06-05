import paddle
import os
import numpy
import math
from utils import (
    check_object_type,
    Const,
    CompareException,
    print_error_log,
    print_warn_log,
    check_file_or_directory_path,
)


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


def gen_data(info, api_name, need_grad, convert_type, real_data_path=None):
    """
    Function Description:
        Based on arg basic information, generate arg data
    Parameter:
        info: arg basic information. Dict
        api_name: API name
        need_grad: set Tensor grad for backward
        convert_type: convert ori_type to dist_type flag.
    """
    check_object_type(info, dict)
    data_type = info.get("type")
    data_path = info.get("real_data_path")
    if data_type in TENSOR_DATA_LIST_PADDLE:
        if data_path and os.path.exists(data_path):
            data = gen_real_tensor(data_path, convert_type)
        else:
            data = gen_random_tensor(info, convert_type)
        data.stop_gradient = False
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
    return data


def gen_real_tensor(data_path, convert_type):
    """
    Function Description:
        Based on API data path, generate input parameters real data
    Parameter:
        data_path: API data path
        convert_type: convert ori_type to dist_type flag.
    """
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
    if convert_type:
        ori_dtype = Const.CONVERT.get(convert_type)[0]
        dist_dtype = Const.CONVERT.get(convert_type)[1]
        if str(data.dtype) == ori_dtype:
            data = data.type(eval(dist_dtype))
    return data


def gen_random_tensor(info, convert_type):
    """
    Function Description:
        Based on API MAX and MIN, generate input parameters random data
    Parameter:
        info: API data info
        convert_type: convert ori_type to dist_type flag.
    """
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
        data = gen_bool_tensor(low, high, shape)
    else:
        data = gen_common_tensor(low_info, high_info, shape, data_dtype, convert_type)
    return data


def fp32_to_hf32_to_fp32(input_tensor):
    # 将输入的float32 tensor转为hf32 tensor，再转为float32 tensor
    input_np = input_tensor.numpy()
    input_int = input_np.view(numpy.int32)
    input_int = numpy.right_shift(numpy.right_shift(input_int, 11) + 1, 1)
    input_int = numpy.left_shift(input_int, 12)
    input_fp32 = input_int.view(numpy.float32)
    input_hf32 = paddle.to_tensor(input_fp32, place=paddle.CPUPlace())
    return input_hf32


def gen_common_tensor(low_info, high_info, shape, data_dtype, convert_type):
    """
    Function Description:
        Based on API basic information, generate int or float tensor
    Parameter:
        low_info: [low, low_origin], low is the minimum value in the tensor removed inf and nan,
        low_origin is the original minimum value in the tensor
        high_info: [high, high_origin], high is the maximum value in the tensor removed inf and nan,
        high_origin is the original maximum value in the tensor
        shape:The shape of Tensor
        data_dtype: The data type of Tensor
        convert_type: convert ori_type to dist_type flag.
    """
    paddle.set_device("cpu")
    if convert_type:
        ori_dtype = Const.CONVERT.get(convert_type)[0]
        if ori_dtype == data_dtype:
            data_dtype = Const.CONVERT.get(convert_type)[1]
    low, low_origin = low_info[0], low_info[1]
    high, high_origin = high_info[0], high_info[1]
    if data_dtype in FLOAT_TYPE_PADDLE:
        if math.isnan(high):
            tensor = paddle.full(
                shape, float("nan"), dtype=eval(REAL_TYPE_PADDLE.get(data_dtype))
            )
            return tensor
        # high_origin为新版json中的属性，只有当high_origin不为None,且high为inf或-inf时，原tensor全为inf或-inf
        if high_origin and high in [float("inf"), float("-inf")]:
            tensor = paddle.full(
                shape, high, dtype=eval(REAL_TYPE_PADDLE.get(data_dtype))
            )
            tensor[-1] = low
            return tensor
        low_scale, high_scale = low, high
        dtype_finfo = paddle.finfo(eval(REAL_TYPE_PADDLE.get(data_dtype)))
        # 适配老版json high和low为inf或-inf的情况，取dtype的最大值或最小值进行放缩
        if high == float("inf"):
            high_scale = dtype_finfo.max
        elif high == float("-inf"):
            high_scale = dtype_finfo.min
        if low == float("inf"):
            low_scale = dtype_finfo.max
        elif low == float("-inf"):
            low_scale = dtype_finfo.min

        scale = high_scale - low_scale
        if data_dtype == "BF16":
            rand01 = paddle.rand(shape, dtype=eval("paddle.float32"))
            tensor = rand01 * scale + low_scale
            tensor = paddle.cast(tensor, dtype="bfloat16")

        else:
            rand01 = paddle.rand(shape, dtype=eval(REAL_TYPE_PADDLE.get(data_dtype)))
            tensor = rand01 * scale + low_scale
    elif (
        "int" in data_dtype
        or "long" in data_dtype
        or "INT" in data_dtype
        or "LONG" in data_dtype
    ):
        low, high = int(low), int(high)
        tensor = paddle.randint(
            low, high + 1, shape, dtype=eval(REAL_TYPE_PADDLE.get(data_dtype))
        )
    else:
        print_error_log("Dtype is not supported: " + data_dtype)
        raise NotImplementedError()
    if tensor.numel() == 0:
        return tensor
    tmp_tensor = tensor.reshape([-1])
    if high_origin and math.isnan(high_origin):
        if tmp_tensor.numel() <= 2:
            tmp_tensor[0] = float("nan")
            tmp_tensor[-1] = high
        else:
            tmp_tensor[0] = low
            tmp_tensor[1] = float("nan")
            tmp_tensor[-1] = high
    else:
        tmp_tensor[0] = low
        tmp_tensor[-1] = high
        if high_origin in [float("inf"), float("-inf")]:
            tmp_tensor[-1] = high_origin
        if low_origin in [float("inf"), float("-inf")]:
            tmp_tensor[0] = low_origin
    data = tmp_tensor.reshape(shape)
    if data_dtype == "BF16":
        data = paddle.cast(data, dtype="bfloat16")
    return data


def gen_bool_tensor(low, high, shape):
    """
    Function Description:
        Based on API basic information, generate bool tensor
    Parameter:
        low: The minimum value in Tensor
        high: The max value in Tensor
        shape:The shape of Tensor
    """
    low, high = int(low), int(high)
    if low > high:
        low, high = high, low
    tensor = paddle.randint(low, high + 1, shape)
    if isinstance(tensor, int):
        data = tensor > 0
    else:
        data = paddle.greater_than(tensor, paddle.to_tensor(0))
    return data


def gen_args(
    args_info, api_name, need_grad=True, convert_type=None, real_data_path=None
):
    """
    Function Description:
        Based on API basic information, generate input parameters: args, for API forward running
    Parameter:
        api_info: API basic information. List
        api_name: API name
        need_grad: set Tensor grad for backward
        convert_type: convert ori_type to dist_type flag.
        real_data_path: the root directory for storing real data.
    """
    check_object_type(args_info, list)
    args_result = []
    for arg in args_info:
        if isinstance(arg, (list, tuple)):
            data = gen_args(arg, api_name, need_grad, convert_type, real_data_path)
        elif isinstance(arg, dict):
            data = gen_data(arg, api_name, need_grad, convert_type, real_data_path)
        elif isinstance(arg, str):
            data = eval(arg)
        else:
            print_warn_log(f"Warning: {arg} is not supported")
            raise NotImplementedError()
        args_result.append(data)
    return args_result


def gen_kwargs(api_info, convert_type=None, real_data_path=None):
    """
    Function Description:
        Based on API basic information, generate input parameters: kwargs, for API forward running
    Parameter:
        api_info: API basic information. Dict
        convert_type: convert ori_type to dist_type flag.
        real_data_path: the root directory for storing real data.
    """
    check_object_type(api_info, dict)
    kwargs_params = api_info.get("kwargs")
    for key, value in kwargs_params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            kwargs_params[key] = gen_list_kwargs(value, convert_type, real_data_path)
        elif isinstance(value, str):
            kwargs_params[key] = eval(value)
        elif value.get("type") in TENSOR_DATA_LIST_PADDLE or value.get(
            "type"
        ).startswith("numpy"):
            kwargs_params[key] = gen_data(value, True, convert_type, real_data_path)
        elif value.get("type") in PADDLE_TYPE:
            gen_paddle_kwargs(kwargs_params, key, value)
        else:
            kwargs_params[key] = value.get("value")
    return kwargs_params


def gen_paddle_kwargs(kwargs_params, key, value):
    if value.get("type") != "paddle.CPUPlace":
        kwargs_params[key] = eval(value.get("value"))


def gen_list_kwargs(kwargs_item_value, convert_type, real_data_path=None):
    """
    Function Description:
        When kwargs value is list, generate the list of kwargs result
    Parameter:
        kwargs_item_value: kwargs value before to generate. List
        convert_type: convert ori_type to dist_type flag.
    """
    kwargs_item_result = []
    for item in kwargs_item_value:
        if item.get("type") in TENSOR_DATA_LIST_PADDLE:
            item_value = gen_data(item, False, convert_type, real_data_path)
        else:
            item_value = item.get("value")
        kwargs_item_result.append(item_value)
    return kwargs_item_result


def gen_api_params(
    api_info, api_name, need_grad=True, convert_type=None, real_data_path=None
):
    """
    Function Description:
        Based on API basic information, generate input parameters: args, kwargs, for API forward running
    Parameter:
        api_info: API basic information. Dict
        api_name: API name
        need_grad: set grad for backward
        convert_type: convert ori_type to dist_type flag.
    """
    check_object_type(api_info, dict)
    if convert_type and convert_type not in Const.CONVERT:
        error_info = f"convert_type params not support {convert_type}."
        raise CompareException(CompareException.INVALID_PARAM_ERROR, error_info)
    kwargs_params = gen_kwargs(api_info, convert_type, real_data_path)
    if api_info.get("args"):
        args_params = gen_args(
            api_info.get("args"), api_name, need_grad, convert_type, real_data_path
        )
    else:
        print_warn_log(f"Warning: No args in {api_info} ")
        args_params = []
    return args_params, kwargs_params
