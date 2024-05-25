import json
import os
import paddle
import paddle.nn.functional as F
from utils import Const, print_warn_log, api_info_preprocess, get_json_contents
from data_generate import gen_api_params, gen_args
from run_ut_utils import hf_32_standard_api, Backward_Message


current_device = 'npu'

not_raise_dtype_set = {'type_as'}
not_detach_set = {'resize_', 'resize_as_', 'set_', 'transpose_', 't_', 'squeeze_', 'unsqueeze_'}
not_backward_list = ['repeat_interleave']


def deal_detach(arg, to_detach=True):
    return arg.detach() if to_detach else arg


def raise_bench_data_dtype(api_name, arg, raise_dtype=None):
    '''
    将标杆数据的dtype转换为raise_dtype
    输入：
        api_name：api名称
        arg：标杆输入
        raise_dtype：需要转换的dtype
    输出： 
        arg: 转换dtype的标杆输入
    '''
    if api_name in hf_32_standard_api and arg.dtype == paddle.float32:
        return arg
    if raise_dtype is None or arg.dtype not in Const.RAISE_PRECISION_PADDLE or raise_dtype == arg.dtype:
        return arg
    return arg.type(raise_dtype)


def retain_grad(tensor):
    def hook(grad):
        tensor.grad = grad
    tensor.register_hook(hook)


def generate_cpu_params(input_args, input_kwargs, need_backward, api_name):
    def recursive_arg_to_cpu(arg_in, to_detach, raise_dtype=None):
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_cpu(arg, to_detach, raise_dtype=raise_dtype) for arg in arg_in)
        elif isinstance(arg_in, paddle.Tensor):
            if need_backward and not arg_in.stop_gradient:
                arg_in = deal_detach(raise_bench_data_dtype(api_name, arg_in.clone(), raise_dtype), to_detach)
                arg_in.stop_gradient = False
                temp_arg_in = arg_in * 1
                arg_in = temp_arg_in.astype(arg_in.dtype)
                retain_grad(arg_in)
                return arg_in
            else:
                return deal_detach(raise_bench_data_dtype(api_name, arg_in.clone(), raise_dtype=raise_dtype), to_detach)
        else:
            return arg_in

    def is_tensor_with_raise_precision(arg_in, check_kwargs=False):
        if arg_in.dtype in Const.RAISE_PRECISION_PADDLE:
            return True
        if check_kwargs and arg_in.dtype in [paddle.float16, paddle.bfloat16]:
            return True
        return False

    def recursive_find_dtypes(arg_in, kwargs=None, check_kwargs=False):
        if isinstance(arg_in, (list, tuple)):
            return set().union(*tuple(recursive_find_dtypes(arg, kwargs, check_kwargs=check_kwargs) for arg in arg_in))
        elif isinstance(arg_in, paddle.Tensor) and is_tensor_with_raise_precision(arg_in, check_kwargs):
            return set([arg_in.dtype])
        elif isinstance(arg_in, dict) and check_kwargs:
            return set().union(*tuple(recursive_find_dtypes(v, kwargs, check_kwargs=True) for v in arg_in.values()))
        return set()

    raise_dtype = None
    need_raise_dtypes = recursive_find_dtypes(input_args)
    need_raise_dtypes.update(recursive_find_dtypes(input_kwargs, check_kwargs=True))
    if len(need_raise_dtypes) == 1:
        raise_dtype = Const.RAISE_PRECISION_PADDLE.get(need_raise_dtypes.pop(), paddle.float32)
    elif len(need_raise_dtypes) >= 2:
        raise_dtype = paddle.float32

    raise_dtype = None if api_name in not_raise_dtype_set else raise_dtype
    is_detach = api_name not in not_detach_set
    cpu_args = recursive_arg_to_cpu(input_args, is_detach, raise_dtype=raise_dtype)
    cpu_kwargs = {key: recursive_arg_to_cpu(value, key != "out" and is_detach, raise_dtype=raise_dtype) for key, value in input_kwargs.items()}
    return cpu_args, cpu_kwargs


def generate_device_params(input_args, input_kwargs, need_backward, api_name):
    def recursive_arg_to_device(arg_in, to_detach):
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_device(arg, to_detach) for arg in arg_in)
        elif isinstance(arg_in, paddle.Tensor):
            if need_backward and not arg_in.stop_gradient:
                arg_in = deal_detach(arg_in.clone(), to_detach).to(current_device)
                arg_in.stop_gradient = False
                temp_arg_in = arg_in * 1
                arg_in = temp_arg_in.astype(arg_in.dtype)
                retain_grad(arg_in)
                return arg_in
            else:
                return deal_detach(arg_in.clone(), to_detach).to(current_device)
        else:
            return arg_in

    is_detach = api_name not in not_detach_set
    device_args = recursive_arg_to_device(input_args, is_detach)
    device_kwargs = \
        {key: recursive_arg_to_device(value, key != "out" and is_detach) for key, value in input_kwargs.items()}
    return device_args, device_kwargs


def run_paddle_api(api_full_name, real_data_path, backward_content, api_info_dict):
    in_fwd_data_list = []
    backward_message = ''
    [api_type, api_name, _] = api_full_name.split('*')
    args, kwargs, need_grad = get_api_info(api_info_dict, api_name, real_data_path)
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    # need_backward = api_full_name in backward_content
    need_backward = True
    if not need_grad:
        print_warn_log(f"{api_full_name} {Backward_Message.UNSUPPORT_BACKWARD_MESSAGE.format(api_full_name)}")
        backward_message += Backward_Message.UNSUPPORT_BACKWARD_MESSAGE
    if api_name in not_backward_list:
        need_grad = False
        print_warn_log(f"{api_full_name} {Backward_Message.NO_BACKWARD_RESULT_MESSAGE.format(api_full_name)}")
        backward_message += Backward_Message.NO_BACKWARD_RESULT_MESSAGE
    need_backward = need_backward and need_grad
    if kwargs.get("device"):
        del kwargs["device"]
    cpu_args, cpu_kwargs = generate_cpu_params(args, kwargs, need_backward, api_name)
    device_args, device_kwargs = generate_device_params(args, kwargs, need_backward, api_name)
    bench_grad_out, device_grad_out = None, None
    out = exec(api_name, api_type)(*cpu_args, **cpu_kwargs)
    device_out = exec(api_name, api_type)(*device_args, **device_kwargs)

    # print('*'*200)
    # print(out)
    # # print(cpu_args)
    # print('*'*200)
    # # print(device_args)
    # print(device_out)

    # current_path = os.path.dirname(os.path.realpath(__file__))
    # ut_setting_path = os.path.join(current_path, "torch_ut_setting.json")
    # api_setting_dict = get_json_contents(ut_setting_path)
    # grad_input_index = api_setting_dict.get(api_name)
    # grad_index = None
    grad, bench_grad = None, None
    # if grad_input_index is not None:
    #     grad_index = grad_input_index.get('grad_index')
    #
    # if need_backward:
    #     if need_to_backward(grad_index, out):
    #         backward_args = backward_content[api_full_name]
    #         grad = gen_args(backward_args, api_name, real_data_path=real_data_path)[0]
    #         bench_grad, _ = generate_cpu_params(grad, {}, False, api_name)
    #         bench_grad_out = run_backward(cpu_args, bench_grad, grad_index, out)
    #         device_grad = grad.clone().detach().to(current_device)
    #         device_grad_out = run_backward(device_args, device_grad, grad_index, device_out)
    #     else:
    #         backward_message += Backward_Message.MULTIPLE_BACKWARD_MESSAGE

    return UtDataInfo(bench_grad_out, device_grad_out, device_out, out, bench_grad, in_fwd_data_list, backward_message)


def get_api_info(api_info_dict, api_name, real_data_path):
    convert_type, api_info_dict = api_info_preprocess(api_name, api_info_dict)
    need_grad = True
    if api_info_dict.get("kwargs") and "out" in api_info_dict.get("kwargs"):
        need_grad = False
    args, kwargs = gen_api_params(api_info_dict, api_name, need_grad, convert_type, real_data_path)
    return args, kwargs, need_grad


def need_to_backward(grad_index, out):
    if grad_index is None and isinstance(out, (list, tuple)):
        return False
    return True


def run_backward(args, grad, grad_index, out):

    if grad_index is not None:
        out[grad_index].backward(grad)
    else:
        out.backward(grad)
    args_grad = []
    for arg in args:
        if isinstance(arg, paddle.Tensor):
            args_grad.append(arg.grad)
    grad_out = args_grad

    return grad_out


def exec(op_name, api_type):
    if "unction" in api_type:
        return getattr(F, op_name)
    elif "addle" in api_type:
        return getattr(paddle, op_name)
    elif "Tensor" in api_type:
        return getattr(paddle.Tensor, op_name)
    else:
        print("In Exec: Undefined api type!")


class UtDataInfo:
    def __init__(self, bench_grad, device_grad, device_output, bench_output, grad_in, in_fwd_data_list,
                 backward_message, rank=0):
        self.bench_grad = bench_grad
        self.device_grad = device_grad
        self.device_output = device_output
        self.bench_output = bench_output
        self.grad_in = grad_in
        self.in_fwd_data_list = in_fwd_data_list
        self.backward_message = backward_message
        self.rank = rank



# if __name__ == "__main__":
#     tensor_x = paddle.randn([8192, 5120])
#     tensor_x.to("bfloat16")
#
#     args = [tensor_x]
#     kwargs = {"axes": [0], "starts": [1024], "ends": [2048]}
#     # 传入方法名称和分类，有paddle、functional、tensor 大小写不区分.
#     ret = exec("slice", "paddle")(*args, **kwargs)
#     print(ret.shape)


if __name__ == '__main__':
    json_path = r'./dump.json'
    with open(json_path, 'r') as json_f:
        data = json.load(json_f)
        for key, value in data.items():
            api_full_name = key
            api_info_dict = value
            real_data_path = ''
            backward_content = ''
            result = run_paddle_api(api_full_name, real_data_path, backward_content, api_info_dict)
            print(result)
