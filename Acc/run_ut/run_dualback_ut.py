from utils import print_info_log
import argparse
import os
import sys
import time
import gc
from tqdm import tqdm
import paddle
import paddle.nn.functional as F
from utils import (
    print_warn_log,
    api_info_preprocess,
    get_json_contents,
    create_directory,
    print_error_log,
    check_path_before_create,
    seed_all,
)
from data_generate import gen_api_params
from run_ut_utils import Backward_Message
from file_check_util import (
    FileCheckConst,
    FileChecker,
    check_link,
    check_file_suffix,
)

# 不需要反向的算子 reshape_ 为动态shape时会报错 算子输出不是leaf tensor.
NO_BACKWARD_OP = ["reshape_"]

# 分组反向的算子，输出为tensor 序列，使用tensor.mean进行均值加和再backward.
Group_Backward_OP = ["slice", "split"]

seed_all()
current_time = time.strftime("%Y%m%d%H%M%S")

tqdm_params = {
    "smoothing": 0,  # 平滑进度条的预计剩余时间，取值范围0到1
    "desc": "Processing",  # 进度条前的描述文字
    "leave": True,  # 迭代完成后保留进度条的显示
    "ncols": 75,  # 进度条的固定宽度
    "mininterval": 0.1,  # 更新进度条的最小间隔秒数
    "maxinterval": 1.0,  # 更新进度条的最大间隔秒数
    "miniters": 1,  # 更新进度条之间的最小迭代次数
    "ascii": None,  # 根据环境自动使用ASCII或Unicode字符
    "unit": "it",  # 迭代单位
    "unit_scale": True,  # 自动根据单位缩放
    "dynamic_ncols": True,  # 动态调整进度条宽度以适应控制台
    "bar_format": "{l_bar}{bar}| {n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",  # 自定义进度条输出
}

def generate_device_params(input_args, input_kwargs, need_backward, api_name):
    current_device = paddle.device.get_device()
    device = current_device[:3]
    def recursive_arg_to_device(arg_in):
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_device(arg) for arg in arg_in)
        elif isinstance(arg_in, paddle.Tensor):
            if "gpu" in current_device:
                arg_in = arg_in.cuda()
            else:
                arg_in = arg_in.to(device)
            if need_backward:
                arg_in.stop_gradient = False
            else:
                arg_in.stop_gradient = True
            return arg_in
        else:
            return arg_in

    device_args = recursive_arg_to_device(input_args)
    device_kwargs = {
        key: recursive_arg_to_device(value) for key, value in input_kwargs.items()
    }
    return device_args, device_kwargs


def _run_ut_save(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    _run_ut_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    run_ut_command_save(args)


def run_ut_command_save(cfg):
    check_link(cfg.forward_input_file)
    forward_file = os.path.realpath(cfg.forward_input_file)
    check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
    out_path = os.path.realpath(cfg.out_path) if cfg.out_path else "./"
    check_path_before_create(out_path)
    create_directory(out_path)
    out_path_checker = FileChecker(
        out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE
    )
    out_path = out_path_checker.common_check()
    forward_content = {}
    if cfg.forward_input_file:
        check_link(cfg.forward_input_file)
        forward_file = os.path.realpath(cfg.forward_input_file)
        check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
        forward_content = get_json_contents(forward_file)

    run_ut_save(forward_content, cfg.real_data_path, out_path, cfg.backend)


def run_ut_save(forward_content, real_data_path, out_path, backend):
    print_info_log("start UT save")
    for i, (api_full_name, api_info_dict) in enumerate(
        tqdm(forward_content.items(), **tqdm_params)
    ):
        Warning_list = []
        try:
            print(api_full_name)
            run_paddle_api_save(
                api_full_name,
                real_data_path,
                api_info_dict,
                out_path,
                backend,
                Warning_list,
            )
            print("*" * 100)
        except Exception as err:
            [_, api_name, _] = api_full_name.split("*")
            if "expected scalar type Long" in str(err):
                print_warn_log(
                    f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                    f"'int32_to_int64' list in accuracy_tools/api_accuracy_check/common/utils.py file"
                )
            else:
                print_error_log(f"Run {api_full_name} UT Error: %s" % str(err))
        finally:
            gc.collect()
    device_str = paddle.device.get_device()
    output_folder = device_str[0:3] + "_output"
    output_dir = os.path.abspath(os.path.join(out_path, output_folder))
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "warning_log.txt")
    File = open(filename, "w")
    for item in Warning_list:
        File.write(item + "\n")


def run_paddle_api_save(
    api_full_name, real_data_path, api_info_dict, dump_path, backend, Warning_list
):
    in_fwd_data_list = []
    backward_message = ""
    [api_type, api_name, _] = api_full_name.split("*")
    args, kwargs, need_grad = get_api_info(api_info_dict, api_name, real_data_path)
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    if not need_grad:
        print_warn_log(
            f"{api_full_name} {Backward_Message.UNSUPPORT_BACKWARD_MESSAGE.format(api_full_name)}"
        )
        backward_message += Backward_Message.UNSUPPORT_BACKWARD_MESSAGE
    paddle.set_device(backend)
    temp = paddle.to_tensor([4])
    del temp
    need_backward = True
    if api_name in NO_BACKWARD_OP:
        need_backward = False
        msg = f"API {api_name} skipped BP, according to 'NO_BACKWARD_OP'!"
        Warning_list.append(msg)
    device_args, device_kwargs = generate_device_params(
        args, kwargs, need_backward, api_name
    )
    device_out = exec(api_name, api_type)(*device_args, **device_kwargs)

    device_str = paddle.device.get_device()
    output_folder = device_str[0:3] + "_output"
    output_dir = os.path.abspath(os.path.join(dump_path, output_folder))
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + "/" + f"{api_full_name}"
    out = device_out
    paddle.save(out, output_path)

    if api_name in Group_Backward_OP:
        msg = f"API:{api_name} has multi outputs, we use .mean() to reduce outputs, and require backwards."
        Warning_list.append(msg)
        print_warn_log(msg)
        try:
            device_grad_out = []
            temp_res = 0
            for out in device_out:
                if isinstance(out, paddle.Tensor):
                    temp_res += out.mean()
            out.backward()
            for arg in device_args:
                if isinstance(arg, paddle.Tensor):
                    device_grad_out.append(arg.grad)
                    output_dir = os.path.abspath(
                        os.path.join(dump_path, output_folder + "_backward")
                    )
                os.makedirs(output_dir, exist_ok=True)
                output_path = output_dir + "/" + f"{api_full_name}"
                paddle.save(device_grad_out, output_path)
        except Exception as err:
            [_, api_name, _] = api_full_name.split("*")
            msg = f"Run API {api_name} backward Error: %s" % str(err)
            print_warn_log(msg)
            Warning_list.append(msg)
        return

    if api_name not in NO_BACKWARD_OP:
        try:
            device_out.backward()
            device_grad_out = []
            for arg in device_args:
                if isinstance(arg, paddle.Tensor):
                    device_grad_out.append(arg.grad)
            output_dir = os.path.abspath(
                os.path.join(dump_path, output_folder + "_backward")
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir + "/" + f"{api_full_name}"
            paddle.save(device_grad_out, output_path)
        except Exception as err:
            [_, api_name, _] = api_full_name.split("*")
            print_warn_log(f"Run API {api_name} backward Error: %s" % str(err))
    else:
        print_warn_log(f"No need to run API {api_name} backward ")
    return


def get_api_info(api_info_dict, api_name, real_data_path):
    paddle.set_device("cpu")
    convert_type, api_info_dict = api_info_preprocess(api_name, api_info_dict)
    need_grad = True
    if api_info_dict.get("kwargs") and "out" in api_info_dict.get("kwargs"):
        need_grad = False
    args, kwargs = gen_api_params(
        api_info_dict, api_name, need_grad, convert_type, real_data_path
    )

    return args, kwargs, need_grad


def need_to_backward(grad_index, out):
    if grad_index is None and isinstance(out, (list, tuple)):
        return False
    return True

def exec(op_name, api_type):
    if "unction" in api_type:
        return getattr(F, op_name)
    elif "addle" in api_type:
        return getattr(paddle, op_name)
    elif "Tensor" in api_type:
        return getattr(paddle.Tensor, op_name)
    else:
        print("In Exec: Undefined api type!")


def _run_ut_parser(parser):
    parser.add_argument(
        "-fwd",
        "--forward",
        dest="forward_input_file",
        default="",
        type=str,
        help="<Optional> The api param tool forward result file: generate from api param tool, "
        "a json file.",
        required=True,
    )
    parser.add_argument(
        "-bp",
        "--backward",
        dest="backward_input_file",
        default="",
        type=str,
        help="<Optional> The api param tool backward result file: generate from api param tool, "
        "a json file.",
        required=False,
    )
    parser.add_argument(
        "-out",
        "--dump_path",
        dest="out_path",
        default="./root/paddlejob/workspace/PaddleAPEX_dump/",
        type=str,
        help="<optional> The ut task result out path.",
        required=False,
    )
    parser.add_argument(
        "--backend",
        dest="backend",
        default="npu",
        type=str,
        help="<optional> The running device NPU or GPU.",
        required=False,
    )
    parser.add_argument(
        "-real_data_path",
        dest="real_data_path",
        nargs="?",
        const="",
        default="",
        type=str,
        help="<optional> In real data mode, the root directory for storing real data "
        "must be configured.",
        required=False,
    )

if __name__ == "__main__":
    _run_ut_save()
    print_info_log("UT save completed")