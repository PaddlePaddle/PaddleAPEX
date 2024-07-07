import argparse
import os
import shutil
import time
import copy
import numpy as np
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


def recursive_delete_arg(arg_in):
    if isinstance(arg_in, (list, tuple)):
        for item in arg_in:
            recursive_delete_arg(item)
        return
    elif isinstance(arg_in, paddle.Tensor):
        del arg_in
        return


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
    fwd_output_dir = os.path.abspath(os.path.join(out_path, dtype_name, "output"))
    bwd_output_dir = os.path.abspath(
        os.path.join(out_path, dtype_name, "output_backward")
    )
    os.makedirs(fwd_output_dir, exist_ok=True)
    os.makedirs(bwd_output_dir, exist_ok=True)
    fwd_output_path = os.path.join(fwd_output_dir, api_call_name)
    bwd_output_path = os.path.join(bwd_output_dir, api_call_name)
    if not isinstance(forward_res, type(None)):
        paddle.save(forward_res, fwd_output_path)
    if not isinstance(backward_res, type(None)):
        paddle.save(backward_res, bwd_output_path)


def evoke_related_test_func(test_mode):
    if test_mode == "acc":
        return run_acc_case
    elif test_mode == "mem":
        return run_mem_case
    elif test_mode == "pro":
        return run_profile_case
    elif test_mode == "all":
        return [run_acc_case, run_mem_case, run_profile_case]
    else:
        raise ValueError("test_mode should be acc, mem, pro or all!")


def ut_case_parsing(forward_content, cfg):
    run_case_funcs = evoke_related_test_func(cfg.test_mode)
    backend = cfg.backend
    out_path = os.path.realpath(cfg.out_path) if cfg.out_path else "./"
    multi_dtype_ut = cfg.multi_dtype_ut.split(",") if cfg.multi_dtype_ut else []
    debug_case = cfg.test_case_name.split(",") if cfg.test_case_name else []
    debug_mode = False
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
                kwargs = {"enforce_dtype": enforce_dtype, "debug_case": debug_case}
                if isinstance(run_case_funcs, list):
                    for run_case in run_case_funcs:
                        run_case(*args, **kwargs)
                else:
                    run_case_funcs(*args, **kwargs)
                print("*" * 100)
        else:
            print(api_call_name)
            args = api_call_name, api_info_dict, backend, out_path
            kwargs = {"enforce_dtype": enforce_dtype, "debug_case": debug_case}
            if isinstance(run_case_funcs, list):
                for run_case in run_case_funcs:
                    run_case(*args, **kwargs)
            else:
                run_case_funcs(*args, **kwargs)
            print("*" * 100)


def create_input_args(api_info, backend, enforce_dtype=None):
    args, kwargs, need_backward = gen_api_params(api_info)
    device_args = recursive_arg_to_device(args, backend, enforce_dtype)
    device_kwargs = {
        key: recursive_arg_to_device(value, backend, enforce_dtype)
        for key, value in kwargs.items()
    }
    return device_args, device_kwargs, need_backward


def create_dout(dout_info_dict, device_out, backend, enforce_dtype=None):
    if dout_info_dict != "Failed":
        dout, _ = gen_args(dout_info_dict)
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
    api_call_name, api_info_dict, backend, out_path, enforce_dtype=None, debug_case=[]
):
    api_info_dict_copy = copy.deepcopy(api_info_dict)
    device_args, device_kwargs, need_backward = create_input_args(
        api_info_dict_copy, backend, enforce_dtype
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
            api_info_dict["dout_list"], device_out, backend, enforce_dtype
        )
        device_grad_out = run_backward(
            api_call_name, device_out, dout, device_args, device_kwargs, need_backward
        )
    except Exception as err:
        msg = "Run_backward Error: %s" % str(err)
        print_warn_log(msg)
        save_tensor(
            device_out, device_grad_out, out_path, api_call_name, enforce_dtype.name
        )
        return
    save_tensor(
        device_out, device_grad_out, out_path, api_call_name, enforce_dtype.name
    )
    return


def get_eps_time(api_call_name):
    api_call_stack = api_call_name.rsplit("*")[0]
    begin = time.time()
    eval(api_call_stack)
    end = time.time()
    return end - begin


def run_profile_case(
    api_call_name, api_info_dict, backend, out_path, enforce_dtype=None, debug_case=[]
):
    print(f"Running {api_call_name} profile test!")
    api_info_dict_copy = copy.deepcopy(api_info_dict)
    device_args, device_kwargs, need_backward = create_input_args(
        api_info_dict_copy, backend, enforce_dtype
    )
    if api_call_name in debug_case:
        x = [device_args, device_kwargs]
        out_path = os.path.realpath(out_path) if out_path else "./"
        save_pth = os.path.join(out_path, "input_data", api_call_name)
        paddle.save(x, save_pth)

    # get dout firstly.
    device_out = run_forward(api_call_name, device_args, device_kwargs)
    dout = create_dout(api_info_dict["dout_list"], device_out, backend, enforce_dtype)
    op_forward_time = []
    op_backward_time = []

    def profile_inner_loop_():
        try:
            fwd_start_time = time.time()
            device_out = run_forward(api_call_name, device_args, device_kwargs)
            fwd_end_time = time.time()
            fwd_time = fwd_end_time - fwd_start_time
        except Exception as err:
            msg = "Run_forward Error: %s" % str(err)
            print_warn_log(msg)
            return -1, -1
        try:
            bwd_start_time = time.time()
            paddle.autograd.backward([device_out], dout)
            bwd_end_time = time.time()
            bwd_time = bwd_end_time - bwd_start_time
            del device_out
        except Exception as err:
            msg = "Run_backward Error: %s" % str(err)
            print_warn_log(msg)
            return fwd_time, -1
        return fwd_time, bwd_time

    try:
        for i in range(110):
            fwd_time, bwd_time = profile_inner_loop_()
            if i >= 10:
                op_forward_time.append(fwd_time)
                op_backward_time.append(bwd_time)
        return fwd_time, bwd_time
    except Exception as err:
        msg = f"Run {api_call_name} profile Error: %s" % str(err)
        print_warn_log(msg)
        Warning_list.append(msg)
        return
    fwd_time = np.array(op_forward_time).mean()
    bwd_time = np.array(op_backward_time).mean()
    log_path = os.path.join(out_path, "profile_analyze.log")
    os.mkdir(out_path) if not os.path.exists(out_path) else None

    eps_time = get_eps_time(api_call_name)
    F = open(log_path, "a")
    op_fwd = api_call_name + "*" + enforce_dtype.name + ".forward"
    op_bwd = api_call_name + "*" + enforce_dtype.name + ".backward"
    F.write(f"{op_fwd}:\t{fwd_time-eps_time}\n")
    F.write(f"{op_bwd}:\t{bwd_time-eps_time}\n")
    F.close()
    return


def run_mem_case(
    api_call_name,
    api_info_dict,
    backend,
    out_path,
    enforce_dtype=None,
    debug_case=[],  # noqa
):
    print(f"Running {api_call_name} mem test!")

    activation_cost = None
    place = framework._current_expected_place_()
    device_id = place.get_device_id()
    before_run_mem = core.device_memory_stat_current_value("Allocated", device_id)
    api_info_dict_copy = copy.deepcopy(api_info_dict)
    device_args, device_kwargs, _ = create_input_args(
        api_info_dict_copy, backend, enforce_dtype
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
    op_name = api_call_name + "*" + enforce_dtype.name
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
        help="<optional> The running device NPU or GPU.",
        required=False,
    )
    parser.add_argument(
        "-enforce-dtype",
        "--dtype",
        dest="multi_dtype_ut",
        default="FP32,FP16,BF16",
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
        default="acc",
        choices=["acc", "pro", "mem", "all"],
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
