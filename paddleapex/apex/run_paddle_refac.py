import argparse
import os
import time
import paddle
import copy
from tqdm import tqdm

from paddleapex.accuracy.utils import (
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


def ut_case_parsing(forward_content, cfg):
    print_info_log("start UT save")
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
                api_info_dict_copy = copy.deepcopy(api_info_dict)
                run_acc_case(
                    api_call_name,
                    api_info_dict_copy,
                    backend,
                    out_path,
                    enforce_dtype,
                    debug_case,
                )
                print("*" * 100)
        else:
            print(api_call_name)
            run_acc_case(api_call_name, api_info_dict, backend, out_path, debug_case)
            print("*" * 100)


def create_input_args(api_info, backend, enforce_dtype=None, debug_case=None):
    args, kwargs, need_backward = gen_api_params(api_info)
    device_args = recursive_arg_to_device(args, backend, enforce_dtype)
    device_kwargs = {
        key: recursive_arg_to_device(value, backend, enforce_dtype)
        for key, value in kwargs.items()
    }
    return device_args, device_kwargs, need_backward


def create_dout(
    dout_info_dict,
    device_out,
    backend,
):
    if dout_info_dict != "Failed":
        dout, _ = gen_args(dout_info_dict)
    else:
        print("dout dump json is None!")
        dout = rand_like(device_out)
    dout = recursive_arg_to_device(dout, backend)
    return dout


def run_forward(
    api_call_name, device_args, device_kwargs, enforce_dtype=None, debug_case=None
):
    api_call_stack = api_call_name.rsplit("*")[0]
    api_name = api_call_stack.rsplit(".")[-1]
    try:
        if api_call_name in debug_case:
            x = [device_args, device_kwargs]
            out_path = os.path.realpath(cfg.out_path) if cfg.out_path else "./"
            save_pth = os.path.join(out_path, "input_data", api_call_name)
            paddle.save(x, save_pth)
        device_out = eval(api_call_stack)(*device_args, **device_kwargs)
        return device_out

    except Exception as err:
        api_name = api_call_name.split("*")[0]
        if enforce_dtype:
            name = api_name + "*" + enforce_dtype.name
            msg = f"Run API{name} Forward Error: %s" % str(err)
        else:
            msg = f"Run API {api_name} Forward Error: %s" % str(err)
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
            api_name = api_call_name.split("*")[0]
            msg = f"Run API {api_name} backward Error: %s" % str(err)
            print_warn_log(msg)
            Warning_list.append(msg)
            return None
    else:
        msg = f"{api_call_name} has no tensor required grad, SKIP Backward"
        print_warn_log(msg)
        Warning_list.append(msg)
        return None


def run_acc_case(
    api_call_name, api_info_dict, backend, out_path, enforce_dtype=None, debug_case=None
):
    device_args, device_kwargs, need_backward = create_input_args(
        api_info_dict, backend, enforce_dtype, debug_case
    )
    try:
        device_out = run_forward(
            api_call_name, device_args, device_kwargs, enforce_dtype, debug_case
        )
    except Exception as err:
        msg = "Run_forward Error: %s" % str(err)
        print_warn_log(msg)
        return None, None

    try:
        device_grad_out = []
        dout = create_dout(api_info_dict["dout_list"], device_out, backend)
        device_grad_out = run_backward(
            api_call_name, device_out, dout, device_args, device_kwargs, need_backward
        )
    except Exception as err:
        msg = "Run_backward Error: %s" % str(err)
        print_warn_log(msg)
        return device_out, None
    save_tensor(
        device_out, device_grad_out, out_path, api_call_name, enforce_dtype.name
    )
    return


def run_profile_case(
    api_call_name, api_info_dict_copy, backend, enforce_dtype, debug_case
):
    pass


def run_mem_case(api_call_name, api_info_dict_copy, backend, enforce_dtype, debug_case):
    pass


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
        help="",
        required=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg_parser(parser)
    cfg = parser.parse_args()
    forward_content = api_json_read(cfg.json_path)
    out_path = os.path.realpath(cfg.out_path) if cfg.out_path else "./"
    ut_case_parsing(forward_content, cfg)
    print_info_log("UT save completed")
    warning_log_pth = os.path.join(out_path, "./warning_log.txt")
    File = open(warning_log_pth, "w")
    for item in Warning_list:
        File.write(item + "\n")
    File.close()
