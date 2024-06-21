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
import torch
# 不需要反向的算子 reshape_ 为动态shape时会报错 算子输出不是leaf tensor.
NO_BACKWARD_OP = ["reshape_", "scale_", "multiply_", "zero_", "zeros", "add_", "expand_as", "scatter_nd"]

SKIP_OP = ["Paddle*maximum","Tensor*__rsub__","Tensor*scale_","Functional*embedding","Paddle*triu", "Tensor*add_",
"Functional*linear","Tensor*multiply_","Tensor*scale","Tensor*zero_","Paddle*slice","Tensor*expand_as",
"Paddle*stack","Functional*silu","Paddle*randn","Tensor*numel","Paddle*concat","Paddle*tril","Paddle*empty"]


# 分组反向的算子，输出为tensor 序列，使用tensor.mean进行均值加和再backward.
Group_Backward_OP = ["torch.split","torch.Tensor.unbind"]

# TORCH_UNSUPPORT_OP = ["Tensor*scale", "Tensor*scale_", "Tensor*numel"]


current_time = time.strftime("%Y%m%d%H%M%S")
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6,7,8"
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

Warning_list = []

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
        process = api_full_name + "<--->" + api_info_dict["origin_paddle_op"]
        paddle_api = api_info_dict["origin_paddle_op"].split("*")[0] +"*"+api_info_dict["origin_paddle_op"].split("*")[1]
        if paddle_api in SKIP_OP:
            print(paddle_api,"  skipped!")
            continue
        try:
            seed_all()
            print(process)
            run_paddle_api_save(
                api_full_name,
                real_data_path,
                api_info_dict,
                out_path,
                backend,
            )
            print("*" * 100)
                        
        except Exception as err:
            import time
            time.sleep(1)
            msg =  str(err) + process
            Warning_list.append(msg)
            
            api_name = api_full_name.split("*")[0]
            if "expected scalar type Long" in str(err):
                print_warn_log(
                    f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                    f"'int32_to_int64' list in accuracy_tools/api_accuracy_check/common/utils.py file"
                )
            else:
                print_error_log(f"Run {api_full_name} UT Error: %s" % str(err))
        finally:
            gc.collect()

    output_folder = "output"
    output_dir = os.path.abspath(os.path.join(out_path, output_folder))
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "../warning_log.txt")
    File = open(filename, "a")
    for item in Warning_list:
        File.write(item + "\n")


def check_none(tensor_list):
    for item in tensor_list:
        assert item is not None, "grad_list has None!"

        

def run_paddle_api_save(
    api_full_name, real_data_path, api_info_dict, dump_path, backend
):
    backward_message = ""
    origin_paddle_name = api_info_dict["origin_paddle_op"]
    del api_info_dict["origin_paddle_op"]
    api_call_stack = api_full_name.split("*")[0]

    args, kwargs = gen_api_params(api_info_dict, None, True, None, None)

    from run_ut_torch import To_torch, To_paddle
    device_args, device_kwargs = To_torch(args, kwargs)
    func_ptr = eval(api_call_stack)
    # print(origin_paddle_name,": ")
    # for item in device_args:
    #     if isinstance(item, torch.Tensor):
    #         print(item.mean(),item.sum())
    #         input()
    # for k, v in device_kwargs.items():
    #     if isinstance(v, torch.Tensor):
    #         print(v.mean(),v.sum())
    #         input()


    op_name = api_call_stack.split(".")[-1]
    if op_name in NO_BACKWARD_OP:
        with torch.no_grad():
            torch_device_out = func_ptr(*device_args, **device_kwargs)
    else:
        # print(device_args)
        # print(device_kwargs)
        # input("press enter to run forward process!")
        torch_device_out = func_ptr(*device_args, **device_kwargs)

    device_out =  To_paddle(torch_device_out)
    output_folder = "torch_output"
    output_dir = os.path.abspath(os.path.join(dump_path, output_folder))
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + "/" + f"{origin_paddle_name}"
    paddle.save(device_out, output_path)


    device_grad_out = []
    if api_call_stack in Group_Backward_OP:
        msg = f"API:{api_full_name} has multi outputs, we use .mean() to reduce outputs, and require backwards."
        Warning_list.append(msg)
        print_warn_log(msg)
        try:
            temp_res = 0
            for out in torch_device_out:
                if isinstance(out, torch.Tensor):
                    temp_res += out.mean()
            temp_res.backward()
            output_dir = os.path.abspath(
                        os.path.join(dump_path, output_folder + "_backward")
                        )
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir + "/" + f"{origin_paddle_name}"
            device_grad_out = To_paddle(device_grad_out)
            check_none(device_grad_out)
            paddle.save(device_grad_out, output_path)
        except Exception as err:
            api_name = api_full_name.split("*")[0]
            msg = f"Run API {api_name} backward Error: %s" % str(err)
            print_warn_log(msg)
            Warning_list.append(msg)
    elif op_name not in NO_BACKWARD_OP:
        try:
            device_out = torch_device_out.sum()
            device_out.backward()
            for arg in device_args:
                if isinstance(arg, torch.Tensor):
                    device_grad_out.append(arg.grad)
            for k,v in device_kwargs.items():
                if isinstance(v, torch.Tensor):
                    device_grad_out.append(v.grad)
            output_dir = os.path.abspath(
                        os.path.join(dump_path, output_folder + "_backward")
                        )
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir + "/" + f"{origin_paddle_name}"
            device_grad_out = To_paddle(device_grad_out)
            check_none(device_grad_out)
            paddle.save(device_grad_out, output_path)
        except Exception as err:
            api_name = api_full_name.split("*")[0]
            print_warn_log(f"Run API {api_name} backward Error: %s" % str(err))
        
    else:
        print_warn_log(f"No need to run API {api_full_name} backward ")
        Warning_list.append(f"No need to run API {api_full_name} backward ")

def need_to_backward(grad_index, out):
    if grad_index is None and isinstance(out, (list, tuple)):
        return False
    return True



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