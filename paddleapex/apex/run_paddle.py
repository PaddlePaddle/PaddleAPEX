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
from paddle import framework
from paddle.base import core
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
    elif isinstance(arg_in, paddle.Tensor):
        del arg_in
        return
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

def convert_out2fp32(arg_in):
    flag = False
    if isinstance(arg_in, (list, tuple)):
        res = []
        for item in arg_in:
            ret_flag, ret_value = convert_out2fp32(item)
            res.append(ret_value)
            flag = flag or ret_flag
        return flag, res
    elif isinstance(arg_in, paddle.Tensor):
        if arg_in.dtype.name == "BF16":
            arg_in = arg_in.cast("float32")
            flag = True
        return flag, arg_in


def recursive_arg_to_cpu(arg_in):
    if isinstance(arg_in, (list, tuple)):
        res = []
        for item in arg_in:
            res.append(recursive_arg_to_cpu(item))
        return res
    elif isinstance(arg_in, paddle.Tensor):
        arg_in = arg_in.to(
            "cpu"
        )  # avoid using .cpu(), which will cause the gradient to be lost
        return arg_in


def recursive_arg_to_device(arg_in, backend, enforce_dtype=None):
    if isinstance(arg_in, (list, tuple)):
        return type(arg_in)(
            recursive_arg_to_device(arg, backend, enforce_dtype) for arg in arg_in
        )
    elif isinstance(arg_in, paddle.Tensor):
        grad_status = arg_in.stop_gradient
        with paddle.no_grad():
            if "gpu" in backend:
                arg_in = arg_in.cuda()
            if "cpu" in backend:
                arg_in = arg_in.cpu()
                if arg_in.dtype.name == "BF16":
                    arg_in = arg_in.cast("float32")
            else:
                arg_in = arg_in.to(backend)
            if enforce_dtype and arg_in.dtype.name in ["BF16", "FP16", "FP32"]:
                arg_in = arg_in.cast(enforce_dtype)
            arg_in.stop_gradient = grad_status
        return arg_in
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
        fwd_BF16_flag, forward_res = convert_out2fp32(forward_res)
        paddle.save([fwd_BF16_flag, forward_res], fwd_output_path)
    if not isinstance(backward_res, type(None)):
        bwd_BF16_flag, backward_res = convert_out2fp32(backward_res)
        paddle.save([bwd_BF16_flag, backward_res], bwd_output_path)


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
    backend = cfg.backend
    out_path = os.path.realpath(cfg.out_path) if cfg.out_path else "./"
    os.mkdir(out_path) if not os.path.exists(out_path) else None
    multi_dtype_ut = cfg.multi_dtype_ut.split(",") if cfg.multi_dtype_ut else []
    debug_case = cfg.test_case_name.split(",") if cfg.test_case_name else []
    debug_mode = False
    paddle.set_device(cfg.backend)
    if len(debug_case) > 0:
        debug_mode = True
    enforce_types = [type_map[item] for item in multi_dtype_ut]
    for i, (api_call_name, api_info_dict) in enumerate(
        tqdm(forward_content.items(), **tqdm_params)
    ):
        if debug_mode and api_call_name not in debug_case:
            continue
        if len(multi_dtype_ut) > 0:
            for enforce_dtype in enforce_types:
                print(api_call_name + "*" + enforce_dtype.name)
                args = api_call_name, api_info_dict, backend, out_path
                kwargs = {"enforce_dtype": enforce_dtype, "debug_case": debug_case, "real_data_path": cfg.real_data}
                for run_case in run_case_funcs:
                    run_case(*args, **kwargs)
                print("*" * 100)
        else:
            print(api_call_name)
            args = api_call_name, api_info_dict, backend, out_path
            kwargs = {"enforce_dtype": None, "debug_case": debug_case}
            if isinstance(run_case_funcs, list):
                for run_case in run_case_funcs:
                    run_case(*args, **kwargs)
            else:
                run_case_funcs(*args, **kwargs)
            print("*" * 100)


def create_input_args(api_info, backend, enforce_dtype=None, real_data_path=None):
    args, kwargs, need_backward = gen_api_params(api_info, real_data_path)
    device_args = recursive_arg_to_device(args, backend, enforce_dtype)
    device_kwargs = {
        key: recursive_arg_to_device(value, backend, enforce_dtype)
        for key, value in kwargs.items()
    }
    return device_args, device_kwargs, need_backward


def create_dout(dout_info_dict, device_out, backend, enforce_dtype=None, real_data_path=None):
    if dout_info_dict[0] != "Failed":
        dout, _ = gen_args(dout_info_dict, real_data_path)
    else:
        print("dout dump json is None!")
        dout = rand_like(device_out)
    dout = recursive_arg_to_device(dout, backend, enforce_dtype)
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
        if isinstance(arg, paddle.Tensor):
            device_grad_out.append(arg.grad)
        if isinstance(arg, list):  # op: concat/stack
            for x in arg:
                if isinstance(x, paddle.Tensor):
                    device_grad_out.append(x.grad)
    for k, v in kwargs.items():
        if isinstance(v, paddle.Tensor):
            device_grad_out.append(v.grad)
        if isinstance(v, list):  # op: concat/stack
            for x in v:
                if isinstance(x, paddle.Tensor):
                    device_grad_out.append(x.grad)
    return device_grad_out


def run_backward(api_call_name, device_out, dout, args, kwargs, need_backward=None):
    if need_backward:
        try:
            paddle.autograd.backward([device_out], dout)
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
    api_call_name, api_info_dict, backend, out_path, enforce_dtype=None, debug_case=[], real_data_path=None
):
    api_info_dict_copy = copy.deepcopy(api_info_dict)
    device_args, device_kwargs, need_backward = create_input_args(
        api_info_dict_copy, backend, enforce_dtype, real_data_path
    )
    print(f"Running {api_call_name} acc test!")
    if api_call_name in debug_case:
        x = [device_args, device_kwargs]
        out_path = os.path.realpath(out_path) if out_path else "./"
        save_pth = os.path.join(out_path, "input_data", api_call_name)
        paddle.save(x, save_pth)
    try:
        device_out = run_forward(api_call_name, device_args, device_kwargs)
    except Exception as err:
        msg = "Run_forward Error: %s" % str(err)
        print_warn_log(msg)
        return

    try:
        device_grad_out = []
        dout = create_dout(
            api_info_dict["dout_list"], device_out, backend, enforce_dtype, real_data_path
        )
        device_grad_out = run_backward(
            api_call_name, device_out, dout, device_args, device_kwargs, need_backward
        )
    except Exception as err:
        msg = "Run_backward Error: %s" % str(err)
        print_warn_log(msg)
        if enforce_dtype:
            save_tensor(
                device_out, device_grad_out, out_path, api_call_name, enforce_dtype.name
            )
        else:
            save_tensor(device_out, device_grad_out, out_path, api_call_name)
        return
    if enforce_dtype:
        save_tensor(
            device_out, device_grad_out, out_path, api_call_name, enforce_dtype.name
        )
    else:
        save_tensor(device_out, device_grad_out, out_path, api_call_name)
    return


def run_profile_case(
    api_call_name, api_info_dict, backend, out_path, enforce_dtype=None, debug_case=[], real_data_path=None
):
    print(f"Running {api_call_name} profile test!")
    api_info_dict_copy = copy.deepcopy(api_info_dict)
    device_args, device_kwargs, need_backward = create_input_args(
        api_info_dict_copy, backend, enforce_dtype, real_data_path
    )
    if api_call_name in debug_case:
        x = [device_args, device_kwargs]
        out_path = os.path.realpath(out_path) if out_path else "./"
        save_pth = os.path.join(out_path, "input_data", api_call_name)
        paddle.save(x, save_pth)
    # device warmming up
    try:
        device_out = run_forward(api_call_name, device_args, device_kwargs)
        dout = create_dout(
            api_info_dict["dout_list"], device_out, backend, enforce_dtype, real_data_path
        )
        paddle.autograd.backward([device_out], dout)
    except Exception as err:
        msg = "Failed in device warming up: %s" % str(err)
        print_warn_log(msg)
        return
    input_shape1 = get_shape(device_args)
    input_shape2 = get_shape(device_kwargs)
    input_shape_lst = merge_two_lists(input_shape1, input_shape2)
    output_shape_lst = get_shape(device_out)
    def profile_inner_loop_():
        try:
            paddle.device.synchronize()
            fwd_start_time = time.time()
            for _ in range(PROFILE_RUN_TIMES):
                device_out = run_forward(api_call_name, device_args, device_kwargs)
            paddle.device.synchronize()
            fwd_end_time = time.time()
            fwd_time = fwd_end_time - fwd_start_time
            fwd_time = fwd_time * 1000000 / float(PROFILE_RUN_TIMES) # fwd_time is in us
        except Exception as err:
            msg = "Run_forward Error: %s" % str(err)
            print_warn_log(msg)
            return -1, -1
        try:
            if not need_backward:
                return fwd_time, -1
            paddle.device.synchronize()
            bwd_start_time = time.time()
            for _ in range(PROFILE_RUN_TIMES):
                device_out = run_forward(api_call_name, device_args, device_kwargs)
                paddle.autograd.backward([device_out], dout)
            paddle.device.synchronize()
            bwd_end_time = time.time()
            bwd_time = bwd_end_time - bwd_start_time  # bwd_time is in second
            bwd_time = bwd_time * 1000000 / float(PROFILE_RUN_TIMES) # bwd_time is in us
            bwd_time = bwd_time - fwd_time
        except Exception as err:
            msg = "Run_backward Error: %s" % str(err)
            print_warn_log(msg)
            return fwd_time, -1
        return fwd_time, bwd_time

    try:
        fwd_time, bwd_time = profile_inner_loop_()
    except Exception as err:
        msg = f"Run {api_call_name} profile Error: %s" % str(err)
        print_warn_log(msg)
        Warning_list.append(msg)
        return
    log_path = os.path.join(out_path, "profile_analyze.log")

    F = open(log_path, "a")
    dtype = "" if not enforce_dtype else f"*{enforce_dtype.name}"
    op_fwd = api_call_name + dtype + ".forward"
    op_bwd = api_call_name + dtype + ".backward"
    print_info_log(f"{op_fwd}:\t{fwd_time}")
    print_info_log(f"{op_bwd}:\t{bwd_time}")
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
    backend,
    out_path,
    enforce_dtype=None,
    debug_case=[],  # noqa
    real_data_path=None
):
    print(f"Running {api_call_name} mem test!")

    activation_cost = None
    place = framework._current_expected_place_()
    device_id = place.get_device_id()
    before_run_mem = core.device_memory_stat_current_value("Allocated", device_id)
    api_info_dict_copy = copy.deepcopy(api_info_dict)
    device_args, device_kwargs, _ = create_input_args(
        api_info_dict_copy, backend, enforce_dtype, real_data_path
    )
    try:
        device_out = run_forward(api_call_name, device_args, device_kwargs)
        recursive_delete_arg(device_args)
        for _, value in device_kwargs.items():
            recursive_delete_arg(value)
        _ = recursive_arg_to_cpu(device_out)
        after_run_mem = core.device_memory_stat_current_value("Allocated", device_id)
        activation_cost = after_run_mem - before_run_mem

    except Exception as err:
        msg = "Run_forward Error: %s" % str(err)
        print_warn_log(msg)
        return
    log_path = os.path.join(out_path, "memory_analyze.log")
    os.mkdir(out_path) if not os.path.exists(out_path) else None
    F = open(log_path, "a")
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
        default="./paddle/",
        type=str,
        help="<optional> The ut task result out path.",
        required=False,
    )
    parser.add_argument(
        "-backend",
        "--backend",
        dest="backend",
        default="gpu",
        type=str,
        help="<optional> The running device DEVICE or BENCH.",
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
        "-real",
        "--real_data",
        dest="real_data",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg_parser(parser)
    cfg = parser.parse_args()
    forward_content = api_json_read(cfg.json_path)
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
