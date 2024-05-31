from utils import print_info_log
import argparse
import os
import sys
import time
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
from Async_save_data import *
seed_all()
not_raise_dtype_set = {'type_as'}
not_detach_set = {'resize_', 'resize_as_', 'set_', 'transpose_', 't_', 'squeeze_', 'unsqueeze_'}
not_backward_list = ['repeat_interleave']
current_time = time.strftime("%Y%m%d%H%M%S")
RESULT_FILE_NAME = f"accuracy_checking_result_" + current_time + ".csv"
DETAILS_FILE_NAME = f"accuracy_checking_details_" + current_time + ".csv"
RunUTConfig = namedtuple('RunUTConfig', ['forward_content', 'backward_content', 'result_csv_path', 'details_csv_path',
                                         'save_error_data', 'is_continue_run_ut', 'real_data_path', 'out_path'])


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


def generate_device_params(input_args, input_kwargs, need_backward, api_name):
    current_device = paddle.device.get_device()
    def recursive_arg_to_device(arg_in, to_detach):
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_device(arg, to_detach) for arg in arg_in)
        elif isinstance(arg_in, paddle.Tensor):
            if need_backward and not arg_in.stop_gradient:
                if "gpu" in current_device:
                    arg_in = deal_detach(arg_in.clone(), to_detach).cuda()
                    arg_in.stop_gradient = False
                elif "npu" in current_device:
                    arg_in = deal_detach(arg_in.clone(), to_detach).to(current_device)
                    arg_in.stop_gradient = False
                return arg_in
            else:
                if "gpu" in current_device:
                    arg_in = deal_detach(arg_in.clone(), to_detach).cuda()
                elif "npu" in current_device:
                    arg_in = deal_detach(arg_in.clone(), to_detach).to(current_device)
                return arg_in
        else:
            return arg_in

    current_device = paddle.device.get_device()
    is_detach = api_name not in not_detach_set
    device_args = recursive_arg_to_device(input_args, is_detach)
    device_kwargs = \
        {key: recursive_arg_to_device(value, key != "out" and is_detach) for key, value in input_kwargs.items()}
    return device_args, device_kwargs


def _run_ut_save(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    _run_ut_parser(parser)
    args = parser.parse_args(sys.argv[1:])
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
                                args.result_csv_path, args.real_data_path, out_path)
    run_ut_save(run_ut_config)


def run_ut_save(config):
    print_info_log("start UT save")
    for i, (api_full_name, api_info_dict) in enumerate(tqdm(config.forward_content.items(), **tqdm_params)):
        try:
            print(api_full_name)
            run_paddle_api_save(api_full_name, config.real_data_path, api_info_dict, config.out_path)
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


def run_paddle_api_save(api_full_name, real_data_path, api_info_dict, dump_path):
    in_fwd_data_list = []
    backward_message = ''
    [api_type, api_name, _] = api_full_name.split('*')
    args, kwargs, need_grad = get_api_info(api_info_dict, api_name, real_data_path)
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    need_backward = True
    need_grad = True
    if not need_grad:
        print_warn_log(f"{api_full_name} {Backward_Message.UNSUPPORT_BACKWARD_MESSAGE.format(api_full_name)}")
        backward_message += Backward_Message.UNSUPPORT_BACKWARD_MESSAGE
    if api_name in not_backward_list:
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

    output_dir = os.path.abspath(os.path.join(dump_path, output_folder))
    os.makedirs(output_dir, exist_ok=True)

    tensor = device_out.clone()
    output_path = output_dir + '/' + f'{api_full_name}'
    pool.safe_parellel_save(tensor.cpu().detach(), output_path, output_path)
    # paddle.save(device_out, output_path)

    current_path = os.path.dirname(os.path.realpath(__file__))
    ut_setting_path = os.path.join(current_path, "paddle_ut_setting.json")
    api_setting_dict = get_json_contents(ut_setting_path)
    grad_input_index = api_setting_dict.get(api_name)
    grad_index = None
    if grad_input_index is not None:
        grad_index = grad_input_index.get('grad_index')

    device_grad_out = None
    if need_backward:
        if need_to_backward(grad_index, device_out):
            device_grad_out = run_backward(device_args, grad_index, device_out)
    else:
        backward_message += Backward_Message.MULTIPLE_BACKWARD_MESSAGE

    output_dir = os.path.abspath(os.path.join(dump_path, output_folder + "_backward"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + '/' + f'{api_full_name}'
    paddle.save(device_grad_out, output_path)
    # tensor_list = []
    # if isinstance(device_grad_out, (list,tuple)):
    #     for item in device_grad_out:
    #         if isinstance(item, paddle.Tensor):
    #             item = item.cpu().detach()
    #             print(item)
    #             tensor_list.append(item)

    pool.safe_parellel_save(tensor_list, output_path, output_path)
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

    if grad_index is not None:
        out[grad_index].backward()
    else:
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
    parser.add_argument("-o", "--dump_path", dest="out_path", default="./root/paddlejob/workspace/PaddleAPEX_dump/", type=str,
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
    global pool
    pool = ThreadPool()
    _run_ut_save()
    print_info_log("UT save completed")
