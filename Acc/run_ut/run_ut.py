import argparse
import json
import os
import sys
import time
import csv
import gc
import re
from collections import namedtuple
from tqdm import tqdm
import paddle
import paddle.nn.functional as F
from utils import Const, print_warn_log, api_info_preprocess, get_json_contents, print_info_log, create_directory, print_error_log, check_path_before_create, seed_all
from data_generate import gen_api_params, gen_args
from run_ut_utils import hf_32_standard_api, Backward_Message
from file_check_util import FileOpen, FileCheckConst, FileChecker, check_link, change_mode, check_file_suffix
# from compare.compare import Comparator

seed_all()
not_raise_dtype_set = {'type_as'}
not_detach_set = {'resize_', 'resize_as_', 'set_', 'transpose_', 't_', 'squeeze_', 'unsqueeze_'}
not_backward_list = ['repeat_interleave']
current_time = time.strftime("%Y%m%d%H%M%S")
RESULT_FILE_NAME = f"accuracy_checking_result_" + current_time + ".csv"
DETAILS_FILE_NAME = f"accuracy_checking_details_" + current_time + ".csv"
RunUTConfig = namedtuple('RunUTConfig', ['forward_content', 'backward_content', 'result_csv_path', 'details_csv_path',
                                         'save_error_data', 'is_continue_run_ut', 'real_data_path'])

tqdm_params = {
    'smoothing': 0,     # 平滑进度条的预计剩余时间，取值范围0到1
    'desc': 'Processing',   # 进度条前的描述文字
    'leave': True,      # 迭代完成后保留进度条的显示
    'ncols': 75,        # 进度条的固定宽度
    'mininterval': 0.1,     # 更新进度条的最小间隔秒数
    'maxinterval': 1.0,     # 更新进度条的最大间隔秒数
    'miniters': 1,  # 更新进度条之间的最小迭代次数
    'ascii': None,  # 根据环境自动使用ASCII或Unicode字符
    'unit': 'it',   # 迭代单位
    'unit_scale': True,     # 自动根据单位缩放
    'dynamic_ncols': True,  # 动态调整进度条宽度以适应控制台
    'bar_format': '{l_bar}{bar}| {n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'   # 自定义进度条输出
}


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
    return arg.astype(raise_dtype)


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
    # cpu_args.stop_gradient = False
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
                # retain_grad(arg_in)
                return arg_in
            else:
                return deal_detach(arg_in.clone(), to_detach).to(current_device)
        else:
            return arg_in

    current_device = paddle.device.get_device()
    is_detach = api_name not in not_detach_set
    device_args = recursive_arg_to_device(input_args, is_detach)
    device_kwargs = \
        {key: recursive_arg_to_device(value, key != "out" and is_detach) for key, value in input_kwargs.items()}
    return device_args, device_kwargs


def run_ut(config):
    print_info_log("start UT test")
    print_info_log(f"UT task result will be saved in {config.result_csv_path}")
    print_info_log(f"UT task details will be saved in {config.details_csv_path}")
    for i, (api_full_name, api_info_dict) in enumerate(tqdm(config.forward_content.items(), **tqdm_params)):
        try:
            print(api_full_name)
            data_info = run_paddle_api(api_full_name, config.real_data_path, config.backward_content, api_info_dict)
        except Exception as err:
            [_, api_name, _] = api_full_name.split("*")
            if "expected scalar type Long" in str(err):
                print_warn_log(f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                               f"'int32_to_int64' list in accuracy_tools/api_accuracy_check/common/utils.py file")
            else:
                print_error_log(f"Run {api_full_name} UT Error: %s" % str(err))
        finally:
            gc.collect()


def run_paddle_api(api_full_name, real_data_path, backward_content, api_info_dict):
    in_fwd_data_list = []
    backward_message = ''
    [api_type, api_name, _] = api_full_name.split('*')
    args, kwargs, need_grad = get_api_info(api_info_dict, api_name, real_data_path)
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    # need_backward = api_full_name in backward_content
    need_backward = True
    need_grad = True
    if not need_grad:
        print_warn_log(f"{api_full_name} {Backward_Message.UNSUPPORT_BACKWARD_MESSAGE.format(api_full_name)}")
        backward_message += Backward_Message.UNSUPPORT_BACKWARD_MESSAGE
    if api_name in not_backward_list:
        # need_grad = False
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

    current_path = os.path.dirname(os.path.realpath(__file__))
    ut_setting_path = os.path.join(current_path, "paddle_ut_setting.json")
    api_setting_dict = get_json_contents(ut_setting_path)
    grad_input_index = api_setting_dict.get(api_name)
    grad_index = None
    if grad_input_index is not None:
        grad_index = grad_input_index.get('grad_index')

    if need_backward:
        # if need_to_backward(grad_index, out):
        bench_grad_out = run_backward(cpu_args, grad_index, out)
        device_grad_out = run_backward(device_args, grad_index, device_out)
    else:
        backward_message += Backward_Message.MULTIPLE_BACKWARD_MESSAGE

    return UtDataInfo(bench_grad_out, device_grad_out, device_out, out, in_fwd_data_list, backward_message)


def _run_ut_save(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    _run_ut_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    # tmp = ['-forward', './dump.json']
    # args = parser.parse_args(tmp)
    run_ut_command_save(args)


def run_ut_command_save(args):
    check_link(args.forward_input_file)
    forward_file = os.path.realpath(args.forward_input_file)
    check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    check_path_before_create(out_path)
    create_directory(out_path)
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    save_error_data = args.save_error_data
    forward_content = {}
    if args.forward_input_file:
        check_link(args.forward_input_file)
        forward_file = os.path.realpath(args.forward_input_file)
        check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
        forward_content = get_json_contents(forward_file)
    if args.filter_api:
        print_info_log("Start filtering the api in the forward_input_file.")
        forward_content = preprocess_forward_content(forward_content)
        print_info_log("Finish filtering the api in the forward_input_file.")
    backward_content = {}
    if args.backward_input_file:
        check_link(args.backward_input_file)
        backward_file = os.path.realpath(args.backward_input_file)
        check_file_suffix(backward_file, FileCheckConst.JSON_SUFFIX)
        backward_content = get_json_contents(backward_file)
    result_csv_path = os.path.join(out_path, RESULT_FILE_NAME)
    details_csv_path = os.path.join(out_path, DETAILS_FILE_NAME)
    if args.result_csv_path:
        result_csv_path = get_validated_result_csv_path(args.result_csv_path, 'result')
        details_csv_path = get_validated_details_csv_path(result_csv_path)
    if save_error_data:
        if args.result_csv_path:
            time_info = result_csv_path.split('.')[0].split('_')[-1]
            global UT_ERROR_DATA_DIR
            UT_ERROR_DATA_DIR = 'ut_error_data' + time_info
    run_ut_config = RunUTConfig(forward_content, backward_content, result_csv_path, details_csv_path, save_error_data,
                                args.result_csv_path, args.real_data_path)
    run_ut_save(run_ut_config)


def run_ut_save(config):
    print_info_log("start UT save")
    for i, (api_full_name, api_info_dict) in enumerate(tqdm(config.forward_content.items(), **tqdm_params)):
        try:
            print(api_full_name)
            run_paddle_api_save(api_full_name, config.real_data_path, config.backward_content, api_info_dict)
            print("*"*200)
        except Exception as err:
            [_, api_name, _] = api_full_name.split("*")
            if "expected scalar type Long" in str(err):
                print_warn_log(f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                               f"'int32_to_int64' list in accuracy_tools/api_accuracy_check/common/utils.py file")
            else:
                print_error_log(f"Run {api_full_name} UT Error: %s" % str(err))
        finally:
            gc.collect()


def run_paddle_api_save(api_full_name, real_data_path, backward_content, api_info_dict):
    in_fwd_data_list = []
    backward_message = ''
    [api_type, api_name, _] = api_full_name.split('*')
    args, kwargs, need_grad = get_api_info(api_info_dict, api_name, real_data_path)
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    # need_backward = api_full_name in backward_content
    need_backward = True
    need_grad = True
    if not need_grad:
        print_warn_log(f"{api_full_name} {Backward_Message.UNSUPPORT_BACKWARD_MESSAGE.format(api_full_name)}")
        backward_message += Backward_Message.UNSUPPORT_BACKWARD_MESSAGE
    if api_name in not_backward_list:
        # need_grad = False
        print_warn_log(f"{api_full_name} {Backward_Message.NO_BACKWARD_RESULT_MESSAGE.format(api_full_name)}")
        backward_message += Backward_Message.NO_BACKWARD_RESULT_MESSAGE
    need_backward = need_backward and need_grad
    if kwargs.get("device"):
        del kwargs["device"]
    device_args, device_kwargs = generate_device_params(args, kwargs, need_backward, api_name)
    device_out = exec(api_name, api_type)(*device_args, **device_kwargs)

    device_str = paddle.device.get_device()
    if device_str[0:3] == "npu":
        output_folder = "npu_output"
    else:
        output_folder = "gpu_output"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(current_dir, "..", output_folder))
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + '/' + f'{api_full_name}'
    paddle.save(device_out, output_path)

    current_path = os.path.dirname(os.path.realpath(__file__))
    ut_setting_path = os.path.join(current_path, "paddle_ut_setting.json")
    api_setting_dict = get_json_contents(ut_setting_path)
    grad_input_index = api_setting_dict.get(api_name)
    grad_index = None
    if grad_input_index is not None:
        grad_index = grad_input_index.get('grad_index')

    if need_backward:
        # if need_to_backward(grad_index, device_out):
        device_grad_out = run_backward(device_args, grad_index, device_out)
    else:
        backward_message += Backward_Message.MULTIPLE_BACKWARD_MESSAGE

    output_dir = os.path.abspath(os.path.join(current_dir, "..", output_folder + "_backward"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + '/' + f'{api_full_name}'
    paddle.save(device_grad_out, output_path)
    return


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


def run_backward(args, grad_index, out):

    # if grad_index is not None:
    #     out[grad_index].backward()
    # else:
    out.backward()
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


def get_validated_result_csv_path(result_csv_path, mode):
    if mode not in ['result', 'detail']:
        raise ValueError("The csv mode must be result or detail")
    result_csv_path_checker = FileChecker(result_csv_path, FileCheckConst.FILE, ability=FileCheckConst.READ_WRITE_ABLE,
                                          file_type=FileCheckConst.CSV_SUFFIX)
    validated_result_csv_path = result_csv_path_checker.common_check()
    if mode == 'result':
        result_csv_name = os.path.basename(validated_result_csv_path)
        pattern = r"^accuracy_checking_result_\d{14}\.csv$"
        if not re.match(pattern, result_csv_name):
            raise ValueError("When continue run ut, please do not modify the result csv name.")
    return validated_result_csv_path


def get_validated_details_csv_path(validated_result_csv_path):
    result_csv_name = os.path.basename(validated_result_csv_path)
    details_csv_name = result_csv_name.replace('result', 'details')
    details_csv_path = os.path.join(os.path.dirname(validated_result_csv_path), details_csv_name)
    details_csv_path_checker = FileChecker(details_csv_path, FileCheckConst.FILE,
                                           ability=FileCheckConst.READ_WRITE_ABLE, file_type=FileCheckConst.CSV_SUFFIX)
    validated_details_csv_path = details_csv_path_checker.common_check()
    return validated_details_csv_path


def _run_ut_parser(parser):
    parser.add_argument("-f", "--forward", dest="forward_input_file", default="", type=str,
                        help="<Optional> The api param tool forward result file: generate from api param tool, "
                             "a json file.",
                        required=True)
    parser.add_argument("-backward", "--backward", dest="backward_input_file", default="", type=str,
                        help="<Optional> The api param tool backward result file: generate from api param tool, "
                             "a json file.",
                        required=False)
    parser.add_argument("-o", "--dump_path", dest="out_path", default="", type=str,
                        help="<optional> The ut task result out path.",
                        required=False)
    parser.add_argument("--backend", dest="backend", default="", type=str,
                        help="<optional> The running device NPU or GPU.",
                        required=False)
    parser.add_argument("--mode", dest="mode", default="random", type=str,
                        help="<optional> The running mode (real/random).",
                        required=False)
    parser.add_argument('-save_error_data', dest="save_error_data", action="store_true",
                        help="<optional> Save compare failed api output.", required=False)
    parser.add_argument("-j", "--jit_compile", dest="jit_compile", action="store_true",
                        help="<optional> whether to turn on jit compile", required=False)

    class UniqueDeviceAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            unique_values = set(values)
            if len(values) != len(unique_values):
                parser.error("device id must be unique")
            for device_id in values:
                if not 0 <= device_id:
                    parser.error("device id must be greater than or equal to 0")
            setattr(namespace, self.dest, values)

    parser.add_argument("-d", "--device", dest="device_id", nargs='+', type=int,
                        help="<optional> set device id to run ut, must be unique and in range 0-7",
                        default=[0], required=False, action=UniqueDeviceAction)
    parser.add_argument("-csv_path", "--result_csv_path", dest="result_csv_path", default="", type=str,
                        help="<optional> The path of accuracy_checking_result_{timestamp}.csv, "
                             "when run ut is interrupted, enter the file path to continue run ut.",
                        required=False)
    parser.add_argument("-real_data_path", dest="real_data_path", nargs="?", const="", default="", type=str,
                        help="<optional> In real data mode, the root directory for storing real data "
                             "must be configured.",
                        required=False)

def preprocess_forward_content(forward_content):
    processed_content = {}
    base_keys_variants = {}
    arg_cache = {}

    for key, value in forward_content.items():
        base_key = key.rsplit(Const.DELIMITER, 1)[0]

        if key not in arg_cache:
            new_args = value['args']
            new_kwargs = value['kwargs']
            filtered_new_args = [
                {k: v for k, v in arg.items() if k not in ['Max', 'Min']}
                for arg in new_args if isinstance(arg, dict)
            ]
            arg_cache[key] = (filtered_new_args, new_kwargs)

        filtered_new_args, new_kwargs = arg_cache[key]

        if base_key not in base_keys_variants:
            processed_content[key] = value
            base_keys_variants[base_key] = {key}
        else:
            is_duplicate = False
            for variant in base_keys_variants[base_key]:
                existing_args, existing_kwargs = arg_cache[variant]
                if existing_args == filtered_new_args and existing_kwargs == new_kwargs:
                    is_duplicate = True
                    break

            if not is_duplicate:
                processed_content[key] = value
                base_keys_variants[base_key].add(key)

    return processed_content


def _run_ut(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    _run_ut_parser(parser)
    # args = parser.parse_args(sys.argv[1:])
    tmp = ['-forward', './dump.json']
    args = parser.parse_args(tmp)
    run_ut_command(args)


def run_ut_command(args):
    check_link(args.forward_input_file)
    forward_file = os.path.realpath(args.forward_input_file)
    check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    check_path_before_create(out_path)
    create_directory(out_path)
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    save_error_data = args.save_error_data
    forward_content = {}
    if args.forward_input_file:
        check_link(args.forward_input_file)
        forward_file = os.path.realpath(args.forward_input_file)
        check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
        forward_content = get_json_contents(forward_file)
    backward_content = {}
    if args.backward_input_file:
        check_link(args.backward_input_file)
        backward_file = os.path.realpath(args.backward_input_file)
        check_file_suffix(backward_file, FileCheckConst.JSON_SUFFIX)
        backward_content = get_json_contents(backward_file)
    result_csv_path = os.path.join(out_path, RESULT_FILE_NAME)
    details_csv_path = os.path.join(out_path, DETAILS_FILE_NAME)
    if args.result_csv_path:
        result_csv_path = get_validated_result_csv_path(args.result_csv_path, 'result')
        details_csv_path = get_validated_details_csv_path(result_csv_path)
    if save_error_data:
        if args.result_csv_path:
            time_info = result_csv_path.split('.')[0].split('_')[-1]
            global UT_ERROR_DATA_DIR
            UT_ERROR_DATA_DIR = 'ut_error_data' + time_info
    run_ut_config = RunUTConfig(forward_content, backward_content, result_csv_path, details_csv_path, save_error_data,
                                args.result_csv_path, args.real_data_path)
    run_ut(run_ut_config)


class UtDataInfo:
    def __init__(self, bench_grad, device_grad, device_output, bench_output, in_fwd_data_list,
                 backward_message, rank=0):
        self.bench_grad = bench_grad
        self.device_grad = device_grad
        self.device_output = device_output
        self.bench_output = bench_output
        self.in_fwd_data_list = in_fwd_data_list
        self.backward_message = backward_message
        self.rank = rank

if __name__ == "__main__":
    _run_ut()
    print_info_log("UT task completed")
