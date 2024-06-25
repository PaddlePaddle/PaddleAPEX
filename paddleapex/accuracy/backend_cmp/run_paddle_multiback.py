import argparse
import os
import time
import paddle
from tqdm import tqdm
from paddleapex.accuracy.utils import (print_info_log, seed_all, gen_api_params, api_info_preprocess,
                                              api_json_read, rand_like, print_warn_log)

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

def recursive_arg_to_device(arg_in, backend):
    if isinstance(arg_in, (list, tuple)):
        return type(arg_in)(recursive_arg_to_device(arg, backend) for arg in arg_in)
    elif isinstance(arg_in, paddle.Tensor):
        if "gpu" in backend:
            arg_in = arg_in.cuda()
        if "cpu" in backend:
            arg_in = arg_in.cpu()
            if arg_in.dtype.name == "BF16":
                arg_in = arg_in.cast("float32")
        else:
            arg_in = arg_in.to(backend)
        return arg_in
    else:
        return arg_in

def ut_case_parsing(forward_content, cfg, out_path):
    backend = cfg.backend
    print_info_log("start UT save")

    fwd_output_dir = os.path.abspath(os.path.join(out_path, "output"))
    bwd_output_dir = os.path.abspath(os.path.join(out_path, "output_backward"))
    os.makedirs(fwd_output_dir, exist_ok=True)
    os.makedirs(bwd_output_dir, exist_ok=True)
    filename = os.path.join(out_path, "./warning_log.txt")
    for i, (api_call_name, api_info_dict) in enumerate(
        tqdm(forward_content.items(), **tqdm_params)
    ):
        # Reset random seed state.
        seed_all(cfg.seed)
        print(api_call_name)
        
        fwd_res, bp_res = run_api_case(
            api_call_name,
            api_info_dict,
            backend,
            filename
        )
        fwd_output_path = os.path.join(fwd_output_dir, api_call_name)
        bwd_output_path = os.path.join(bwd_output_dir, api_call_name)
        if fwd_res:
            paddle.save(fwd_res, fwd_output_path)
        if bp_res:
            paddle.save(bp_res, bwd_output_path)
        print("*" * 100)

def run_api_case(
    api_call_name, api_info_dict, backend, warning_log_pth
):
    Warning_list = []
    api_call_stack = api_call_name.rsplit("*")[0]
    api_name = api_call_stack.rsplit(".")[-1]
    args, kwargs, need_backward = get_api_info(api_info_dict, api_name)

    if api_name =="scatter_nd":
        return None, None

    ##################################################################
    ##      RUN FORWARD
    ##################################################################
    try:
        device_args = recursive_arg_to_device(args, backend)
        device_kwargs = {
            key: recursive_arg_to_device(value, backend) for key, value in kwargs.items()
        }
        device_out = eval(api_call_stack)(*device_args, **device_kwargs)

    except Exception as err:
        api_name = api_call_name.split("*")[0]
        msg = f"Run API {api_name} Forward Error: %s" % str(err)
        print_warn_log(msg)
        Warning_list.append(msg)
        File = open(warning_log_pth, "a")
        for item in Warning_list:
            File.write(item + "\n")
        File.close()
        return  None, None

    ##################################################################
    ##      RUN BACKWARD
    ##################################################################
    if need_backward:
        try:
            device_grad_out = []
            dout = rand_like(device_out)

            dout = recursive_arg_to_device(dout)
            paddle.autograd.backward(
                [device_out], [dout]
            )
            for arg in device_args:
                if isinstance(arg, paddle.Tensor):
                    device_grad_out.append(arg.grad)
            for _, v in device_kwargs.items():
                if isinstance(v, paddle.Tensor):
                    device_grad_out.append(v.grad)

        except Exception as err:
            api_name = api_call_name.split("*")[0]
            msg = f"Run API {api_name} backward Error: %s" % str(err)
            print_warn_log(msg)
            Warning_list.append(msg)
            return device_out, None
    else:
        msg = f"{api_call_name} has no tensor required grad, SKIP Backward"
        print_warn_log(msg)
        Warning_list.append(msg)
        return device_out, None
    
    File = open(warning_log_pth, "a")
    for item in Warning_list:
        File.write(item + "\n")
    File.close()
    return  device_out, device_grad_out


def get_api_info(api_info_dict, api_name):
    convert_type, api_info_dict = api_info_preprocess(api_name, api_info_dict)
    args, kwargs, need_grad = gen_api_params(
        api_info_dict, convert_type
    )
    return args, kwargs, need_grad

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
        default="./root/paddlejob/workspace/PaddleAPEX_dump/",
        type=str,
        help="<optional> The ut task result out path.",
        required=False,
    )
    parser.add_argument(
        "-backend",
        "--backend",
        dest="backend",
        default="npu",
        type=str,
        help="<optional> The running device NPU or GPU.",
        required=False,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg_parser(parser)
    cfg = parser.parse_args()
    forward_content = api_json_read(cfg.json_path)
    out_path = os.path.realpath(cfg.out_path) if cfg.out_path else "./"
    ut_case_parsing(forward_content, cfg, out_path)
    print_info_log("UT save completed")