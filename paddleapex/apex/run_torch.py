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

import argparse
import os
import shutil
import time
import copy
from tqdm import tqdm
import paddle
import torch
from utils import (
    print_info_log,
    gen_api_params,
    api_json_read,
    check_grad_list,
    rand_like,
    gen_args,
    print_warn_log,
)

type_map = {
    "FP16": paddle.float16,
    "FP32": paddle.float32,
    "BF16": paddle.bfloat16,
}
Warning_list = []

torch_type_map = {
    "FP16": torch.float16,
    "FP32": torch.float32,
    "BF16": torch.bfloat16,
}

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
PROFILE_RUN_TIMES = 100


def recursive_delete_arg(arg_in):
    if isinstance(arg_in, (list, tuple)):
        for item in arg_in:
            recursive_delete_arg(item)
        return
    elif isinstance(arg_in, torch.Tensor):
        del arg_in
        return

string_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
dtype_map = {
    paddle.bfloat16: torch.bfloat16,
    paddle.float16: torch.float16,
    paddle.float32: torch.float32,
}

# convert torch.Tensor to paddle.Tensor & BF16 to FP32
def convert_type(arg_in):
    flag = False
    if isinstance(arg_in, (list, tuple)):
        res = []
        for item in arg_in:
            ret_flag, ret_value = convert_type(item)
            res.append(ret_value)
            flag = flag or ret_flag
        return flag, res
    elif isinstance(arg_in, torch.Tensor):
        if arg_in.dtype == torch.bfloat16:
            convert_arg = arg_in.to(torch.float32).detach().cpu().numpy()
            flag = True
        else:
            convert_arg = arg_in.detach().cpu().numpy()
        convert_arg = paddle.to_tensor(convert_arg)
        return flag, convert_arg
    else:
        raise ValueError("convert_type error")

def get_shape(arg_in):
    if isinstance(arg_in, (list, tuple)):
        res = []
        for item in arg_in:
            ret_value = get_shape(item)
            res.append(ret_value)
        return res
    elif isinstance(arg_in, paddle.Tensor):
        shape = arg_in.shape
        return shape

def merge_two_lists(lst1, lst2):
    merged_list = []
    if lst1 is None and lst2 is not None:
        merged_list = lst2
    elif lst1 is not None and lst2 is None:
        merged_list = lst1
    elif lst1 is None and lst2 is None:
        merged_list = []
    else:
        for item in lst1:
            if item is None:
                continue
            else:
                merged_list.append(item)
        for item in lst2:
            if item is None:
                continue
            else:
                merged_list.append(item)
    return merged_list
def recursive_arg_to_cpu(arg_in):
    if isinstance(arg_in, (list, tuple)):
        res = []
        for item in arg_in:
            res.append(recursive_arg_to_cpu(item))
        return res
    elif isinstance(arg_in, torch.Tensor):
        arg_in = arg_in.cpu()
        return arg_in


# paddle to torch
def recursive_arg_to_device(arg_in, enforce_dtype=None, enforce_grad=False):
    if isinstance(arg_in, (list, tuple)):
        return tuple(
            recursive_arg_to_device(arg, enforce_dtype) for arg in arg_in
        )
    elif isinstance(arg_in, paddle.Tensor) and arg_in.dtype.name in [
        "FP32",
        "FP16",
        "BF16",
    ]:
        grad_status = arg_in.stop_gradient
        if enforce_dtype and enforce_dtype.name == "BF16":
            arg_in = arg_in.cast(paddle.float32).numpy()
            arg_in_torch = torch.from_numpy(arg_in).cuda()
            arg_in_torch = arg_in_torch.to(torch.bfloat16)
        elif enforce_dtype:
            arg_in = arg_in.cast(enforce_dtype).numpy()
            arg_in_torch = torch.from_numpy(arg_in).cuda()
        elif not enforce_dtype and arg_in.dtype.name == "BF16":
            arg_in = arg_in.cast(paddle.float32).numpy()
            arg_in_torch = torch.from_numpy(arg_in).cuda()
            arg_in_torch = arg_in_torch.to(torch.bfloat16)
        else:
            arg_in = arg_in.numpy()
            arg_in_torch = torch.from_numpy(arg_in).cuda()
        try:
            if enforce_grad:
                arg_in_torch.requires_grad = True
            else:
                arg_in_torch.requires_grad = not grad_status
        except AttributeError as err:
            print("In enforce_grad mode: ", str(err))
        return arg_in_torch
    elif isinstance(arg_in, paddle.Tensor) and arg_in.dtype.name not in [
        "FP32",
        "FP16",
        "BF16",
    ]:
        arg_in = arg_in.numpy()
        arg_in_torch = torch.from_numpy(arg_in).cuda()
        try:
            if enforce_grad:
                arg_in_torch.requires_grad = True
        except AttributeError as err:
            print("In enforce_grad mode: ", str(err))
        return arg_in_torch
    else:
        return arg_in


def save_tensor(forward_res, backward_res, out_path, api_call_name, dtype_name=""):
    if dtype_name == "":
        bwd_output_dir = os.path.abspath(os.path.join(out_path, "output_backward"))
        fwd_output_dir = os.path.abspath(os.path.join(out_path, "output"))
    else:
        bwd_output_dir = os.path.abspath(
            os.path.join(out_path, dtype_name, "output_backward")
        )
        fwd_output_dir = os.path.abspath(os.path.join(out_path, dtype_name, "output"))
    fwd_output_path = os.path.join(fwd_output_dir, api_call_name)
    bwd_output_path = os.path.join(bwd_output_dir, api_call_name)
    os.makedirs(fwd_output_dir, exist_ok=True)
    os.makedirs(bwd_output_dir, exist_ok=True)
    if not isinstance(forward_res, type(None)):
        fwd_BF16_flag, forward_res = convert_type(forward_res)
        paddle.save([fwd_BF16_flag, forward_res], fwd_output_path)
    if not isinstance(backward_res, type(None)):
        bwd_BF16_flag, backward_res = convert_type(backward_res)
        paddle.save([bwd_BF16_flag, backward_res], bwd_output_path)


def convert_args2torch_style(arg):
    if isinstance(arg, (list, tuple)):
        return type(arg)(convert_args2torch_style(item) for item in arg)
    elif isinstance(arg, str):
        if arg in string_map:
            return string_map[arg]
        else:
            return arg
    elif isinstance(arg, dict):
        for k,v in arg.items():
            arg[k] = convert_args2torch_style(v)
        return arg
    elif isinstance(arg, paddle.dtype):
        if arg in dtype_map:
            arg = dtype_map[arg]
        return arg
    else:
        return arg


def evoke_related_test_func(test_mode):
    func_method = []
    if "acc" in test_mode:
        func_method.append(run_acc_case)
    if "mem" in test_mode:
        func_method.append(run_mem_case)
    if "pro" in test_mode:
        func_method.append(run_profile_case)
    if test_mode == "all":
        return [run_acc_case, run_mem_case, run_profile_case]
    if len(func_method) == 0:
        raise ValueError("test mode is not supported!")
    return func_method


def ut_case_parsing(forward_content, cfg):
    run_case_funcs = evoke_related_test_func(cfg.test_mode)
    out_path = os.path.realpath(cfg.out_path) if cfg.out_path else "./"
    os.mkdir(out_path) if not os.path.exists(out_path) else None
    multi_dtype_ut = cfg.multi_dtype_ut.split(",") if cfg.multi_dtype_ut else []
    debug_case = cfg.test_case_name.split(",") if cfg.test_case_name else []
    debug_mode = False
    if len(debug_case) > 0:
        debug_mode = True
    enforce_types = [type_map[item] for item in multi_dtype_ut]
    for i, (api_call_name, api_info_dict) in enumerate(
        tqdm(forward_content.items(), **tqdm_params)
    ):
        if "unmatch" in api_call_name:
            print(api_info_dict["origin_paddle_op"], " is not supported")
            continue
        if debug_mode and api_call_name not in debug_case:
            continue
        if len(multi_dtype_ut) > 0:
            for enforce_dtype in enforce_types:
                process = (
                    api_call_name
                    + "*"
                    + enforce_dtype.name
                    + "<--->"
                    + api_info_dict["origin_paddle_op"]
                )
                print(process)
                args = api_call_name, api_info_dict, out_path
                kwargs = {"enforce_dtype": enforce_dtype, "debug_case": debug_case, "real_data_path": cfg.real_data}
                for run_case in run_case_funcs:
                    run_case(*args, **kwargs)
                print("*" * 100)
        else:
            process = api_call_name + "<--->" + api_info_dict["origin_paddle_op"]
            print(process)
            args = api_call_name, api_info_dict, out_path
            kwargs = {"enforce_dtype": None, "debug_case": debug_case}
            if isinstance(run_case_funcs, list):
                for run_case in run_case_funcs:
                    run_case(*args, **kwargs)
            else:
                run_case_funcs(*args, **kwargs)
            print("*" * 100)

def create_dout(dout_info_dict, device_out, enforce_dtype=None, real_data_path=None):
    if dout_info_dict[0] != "Failed":
        dout, _ = gen_args(dout_info_dict, real_data_path)
    else:
        print("dout dump json is None!")
        device_out = convert_type(device_out)[1]
        dout = rand_like(device_out)
    dout = recursive_arg_to_device(dout, enforce_dtype)
    return dout


def run_forward(api_call_name, device_args, device_kwargs):
    api_call_stack = api_call_name.rsplit("*")[0]
    try:
        device_out = eval(api_call_stack)(*device_args, **device_kwargs)
        return device_out
    except Exception as err:
        msg = f"Run API {api_call_name} Forward Error: %s" % str(err)
        print_warn_log(msg)
        Warning_list.append(msg)
        return None


def get_grad_tensor(args, kwargs):
    device_grad_out = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            device_grad_out.append(arg.grad)
        if isinstance(arg, list):  # op: concat/stack
            for x in arg:
                if isinstance(x, torch.Tensor):
                    device_grad_out.append(x.grad)
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            device_grad_out.append(v.grad)
        if isinstance(v, list):  # op: concat/stack
            for x in v:
                if isinstance(x, torch.Tensor):
                    device_grad_out.append(x.grad)
    return device_grad_out


def run_backward(api_call_name, device_out, dout, args, kwargs, need_backward=None):
    if need_backward:
        try:
            if isinstance(device_out, torch.Tensor):
                dout = dout[0].reshape(device_out.shape)
            torch.autograd.backward([device_out], dout)
            device_grad_out = get_grad_tensor(args, kwargs)
            device_grad_out = check_grad_list(device_grad_out)
            if device_grad_out is None:
                msg = f"{api_call_name} grad_list is None"
                Warning_list.append(msg)
            return device_grad_out
        except Exception as err:
            msg = f"Run API {api_call_name} backward Error: %s" % str(err)
            print_warn_log(msg)
            Warning_list.append(msg)
            return None
    else:
        msg = f"{api_call_name} has no tensor required grad, SKIP Backward"
        print_warn_log(msg)
        Warning_list.append(msg)
        return None


def run_acc_case(
    api_call_name, api_info_dict, out_path, enforce_dtype=None, debug_case=[], real_data_path=None
):
    api_info_dict_copy = copy.deepcopy(api_info_dict)
    debug_mode = len(debug_case) > 0
    args, kwargs, need_backward = gen_api_params(api_info_dict_copy, real_data_path)
    args = convert_args2torch_style(args)
    kwargs = convert_args2torch_style(kwargs)
    if debug_mode:
        if api_info_dict["origin_paddle_op"] in debug_case:
            x = [args, kwargs]
            out_path = os.path.realpath(out_path) if out_path else "./"
            save_pth = os.path.join(out_path, "input_data", api_call_name)
            paddle.save(x, save_pth)
    device_args = recursive_arg_to_device(args, enforce_dtype)
    device_kwargs = {
        key: recursive_arg_to_device(value, enforce_dtype)
        for key, value in kwargs.items()
    }
    paddle_name = api_info_dict["origin_paddle_op"]
    print(f"Running {api_call_name} <---> {paddle_name} acc test!")

    try:
        device_out = run_forward(api_call_name, device_args, device_kwargs)
    except Exception as err:
        msg = "Run_forward Error: %s" % str(err)
        print_warn_log(msg)
        return

    try:
        device_grad_out = []
        if need_backward:
            dout = create_dout(api_info_dict["dout_list"], device_out, enforce_dtype, real_data_path)
            device_grad_out = run_backward(
                api_call_name,
                device_out,
                dout,
                device_args,
                device_kwargs,
                need_backward,
            )
        else:
            print_info_log(
                f"{api_call_name} has no tensor required grad, SKIP Backward"
            )
    except Exception as err:
        msg = "Run_backward Error: %s" % str(err)
        print_warn_log(msg)
        type_name = enforce_dtype.name if enforce_dtype else ""
        save_tensor(device_out, device_grad_out, out_path, paddle_name, type_name)
        return
    type_name = enforce_dtype.name if enforce_dtype else ""
    save_tensor(device_out, device_grad_out, out_path, paddle_name, type_name)
    return


def run_profile_case(
    api_call_name, api_info_dict, out_path, enforce_dtype=None, debug_case=[], real_data_path=None
):
    print(f"Running {api_call_name} profile test!")
    api_info_dict_copy = copy.deepcopy(api_info_dict)
    paddle_name = api_info_dict["origin_paddle_op"]
    args, kwargs, need_backward = gen_api_params(api_info_dict_copy, real_data_path)
    input_shape1 = get_shape(device_args)
    input_shape2 = get_shape(device_kwargs)
    input_shape_lst = merge_two_lists(input_shape1, input_shape2)
    output_shape_lst = get_shape(device_out)
    debug_mode = len(debug_case) > 0
    if debug_mode:
        if api_info_dict["origin_paddle_op"] in debug_case:
            x = [args, kwargs]
            out_path = os.path.realpath(out_path) if out_path else "./"
            save_pth = os.path.join(out_path, "input_data", paddle_name)
            paddle.save(x, save_pth)
    device_args = recursive_arg_to_device(args, enforce_dtype)
    device_kwargs = {
        key: recursive_arg_to_device(value, enforce_dtype)
        for key, value in kwargs.items()
    }
    device_args = convert_args2torch_style(device_args)
    device_kwargs = convert_args2torch_style(device_kwargs)
    # device warmming up
    try:
        device_out = run_forward(api_call_name, device_args, device_kwargs)
        # api not support
        if device_out is None:
            msg = "Device warming up failed, it may caused by unsupported operator"
            print_warn_log(msg)
            return
        dout = create_dout(api_info_dict["dout_list"], device_out, enforce_dtype, real_data_path)
        # recognize size([]) and size([1])
        if isinstance(device_out, torch.Tensor):
            if isinstance(dout, (list,tuple)):
                dout = dout[0].reshape(device_out.shape)
            else:
                dout = dout.reshape(device_out.shape)

        torch.autograd.backward([device_out], dout)
    except Exception as err:
        msg = "Failed in device warming up: %s" % str(err)
        print_warn_log(msg)
        return

    def profile_inner_loop_(dout):
        try:
            torch.cuda.synchronize()
            fwd_start_time = time.time()
            for _ in range(PROFILE_RUN_TIMES):
                device_out = run_forward(api_call_name, device_args, device_kwargs)
            torch.cuda.synchronize()
            fwd_end_time = time.time()
            fwd_time = fwd_end_time - fwd_start_time
            fwd_time = fwd_time * 1000000  # fwd_time is in us
        except Exception as err:
            msg = "Run_forward Error: %s" % str(err)
            print_warn_log(msg)
            return -1, -1
        try:
            if not need_backward:
                return fwd_time, -1
            # recognize size([]) and size([1])
            if isinstance(device_out, torch.Tensor):
                if isinstance(dout, list):
                    dout = dout[0].reshape(device_out.shape)
                else:
                    dout = dout.reshape(device_out.shape)
            bwd_start_time = time.time()
            torch.cuda.synchronize()
            for _ in range(PROFILE_RUN_TIMES):
                torch.autograd.backward([device_out], dout, retain_graph=True)
            torch.cuda.synchronize()
            bwd_end_time = time.time()
            bwd_time = bwd_end_time - bwd_start_time  # bwd_time is in second
            bwd_time = bwd_time * 1000000  # bwd_time is in us
        except Exception as err:
            msg = "Run_backward Error: %s" % str(err)
            print_warn_log(msg)
            return fwd_time, -1
        return fwd_time, bwd_time

    try:
        fwd_time, bwd_time = profile_inner_loop_(dout)
    except Exception as err:
        msg = f"Run {api_call_name} profile Error: %s" % str(err)
        print_warn_log(msg)
        Warning_list.append(msg)
        return
    if not enforce_dtype:
        log_path = os.path.join(out_path, "profile_analyze.log")
    else:
        log_path = os.path.join(out_path, enforce_dtype.name, "profile_analyze.log")

    F = open(log_path, "a")
    if enforce_dtype:
        op_fwd = paddle_name + "*" + enforce_dtype.name + ".forward"
        op_bwd = paddle_name + "*" + enforce_dtype.name + ".backward"
    else:
        op_fwd = paddle_name + ".forward"
        op_bwd = paddle_name + ".backward"
    print_info_log(f"{op_fwd}:\t{fwd_time/float(PROFILE_RUN_TIMES)}")
    print_info_log(f"{op_bwd}:\t{bwd_time/float(PROFILE_RUN_TIMES)}")
    dtype = "\t" if not enforce_dtype else f"\t{enforce_dtype.name}"
    msg_fwd = f"{api_call_name}.forward\tdtype{dtype}\tinput shape\t{input_shape_lst}\toutput shape\t{output_shape_lst}\tforward\t{fwd_time}"
    msg_bwd = f"{api_call_name}.backward\tdtype{dtype}\tinput shape\t{input_shape_lst}\toutput shape\t{output_shape_lst}\tbackward\t{bwd_time}"

    F.write(msg_fwd + "\n")
    F.write(msg_bwd + "\n")
    F.close()
    return


def run_mem_case(
    api_call_name,
    api_info_dict,
    out_path,
    enforce_dtype=None,
    debug_case=[],  # noqa
    real_data_path=None
):
    print(f"Running {api_call_name} mem test!")
    paddle_name = api_info_dict["origin_paddle_op"]
    activation_cost = None
    before_run_mem = torch.cuda.memory_allocated()
    api_info_dict_copy = copy.deepcopy(api_info_dict)
    args, kwargs, _ = gen_api_params(api_info_dict_copy, real_data_path)
    args = convert_args2torch_style(args)
    kwargs = convert_args2torch_style(kwargs)
    device_args = recursive_arg_to_device(args, enforce_dtype)
    device_kwargs = {
        key: recursive_arg_to_device(value, enforce_dtype)
        for key, value in kwargs.items()
    }
    try:
        device_out = run_forward(api_call_name, device_args, device_kwargs)
        recursive_delete_arg(device_args)
        for _, value in device_kwargs.items():
            recursive_delete_arg(value)
        _ = recursive_arg_to_cpu(device_out)
        after_run_mem = torch.cuda.memory_allocated()
        activation_cost = after_run_mem - before_run_mem

    except Exception as err:
        msg = "Run_forward Error: %s" % str(err)
        print_warn_log(msg)
        return
    log_path = os.path.join(out_path, "memory_analyze.log")
    os.mkdir(out_path) if not os.path.exists(out_path) else None
    F = open(log_path, "a")
    if enforce_dtype:
        op_name = paddle_name + "*" + enforce_dtype.name + ".forward"
    else:
        op_name = paddle_name + ".forward"
    dtype = "" if not enforce_dtype else f"*{enforce_dtype.name}"
    op_name = api_call_name + dtype + ".forward"
    F.write(f"{op_name}:\t{str(activation_cost)}\n")
    F.close()
    return


def arg_parser(parser):
    parser.add_argument(
        "-json",
        "--json",
        dest="json_path",
        default="",
        type=str,
        help="Dump json file path",
        required=True,
    )
    parser.add_argument(
        "-out",
        "--dump_path",
        dest="out_path",
        default="./torch/",
        type=str,
        help="<optional> The ut task result out path.",
        required=False,
    )
    parser.add_argument(
        "-dtype",
        "--enforce-dtype",
        dest="multi_dtype_ut",
        default="",
        type=str,
        help="",
        required=False,
    )
    parser.add_argument(
        "-op",
        "--op_name",
        dest="test_case_name",
        default="",
        type=str,
        help="debug_op name",
        required=False,
    )
    parser.add_argument(
        "-mode",
        "--mode",
        dest="test_mode",
        default="all",
        type=str,
        help="debug_op name",
        required=False,
    )
    parser.add_argument(
        "-real",
        "--real_data",
        dest="real_data",
        default="",
        type=str,
        help="",
        required=False,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg_parser(parser)
    cfg = parser.parse_args()
    forward_content = api_json_read(cfg.json_path)
    if os.path.realpath(cfg.out_path) == os.path.realpath("./"):
        cfg.out_path = "./torch/"
        print_warn_log("The output path is replaced with \"./torch\" . Please do not use the current directory as the output directory.")
    out_path = os.path.realpath(cfg.out_path) if cfg.out_path else "./"
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    ut_case_parsing(forward_content, cfg)
    print_info_log("UT save completed")
    warning_log_pth = os.path.join(out_path, "./warning_log.txt")
    File = open(warning_log_pth, "w")
    for item in Warning_list:
        File.write(item + "\n")
    File.close()
