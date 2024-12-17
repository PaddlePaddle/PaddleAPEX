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
import os
import csv
import argparse
import sys
import time
import paddle
import tqdm
import pandas as pd

import paddle.distributed as dist

from compare_utils.compare import Comparator
from compare_utils.compare_dependency import print_info_log, FileOpen

current_time = time.strftime("%Y%m%d%H%M%S")
rank = dist.get_rank()

RESULT_FILE_NAME = "accuracy_checking_result_" + current_time + "_" + str(rank) + ".csv"
DETAILS_FILE_NAME = "accuracy_checking_details_" + current_time + "_" + str(rank) + ".csv"

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
    "bar_format": "{l_bar}{bar}| {n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",  # 自定义进度条输出格式
}


def _compare_parser(parser):
    parser.add_argument(
        "-bench",
        "--benchmark",
        dest="bench_dir",
        type=str,
        help="The executed output api tensor path directory on BENCH",
        required=True,
    )
    parser.add_argument(
        "-device",
        "--device",
        dest="device_dir",
        type=str,
        help="The executed output api tensor path directory on DEVICE",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="out_path",
        default="",
        type=str,
        help="<Optional> The result out path",
    )


def compare_command(args):
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    os.makedirs(out_path, exist_ok=True)
    result_csv_path = os.path.join(out_path, RESULT_FILE_NAME)
    details_csv_path = os.path.join(out_path, DETAILS_FILE_NAME)
    print_info_log(f"Compare task result will be saved in {result_csv_path}")
    print_info_log(f"Compare task details will be saved in {details_csv_path}")
    bench_dir = os.path.join(args.bench_dir, "./output")
    device_dir = os.path.join(args.device_dir, "./output")
    bench_back_dir = os.path.join(args.bench_dir, "./output_backward")
    device_back_dir = os.path.join(args.device_dir, "./output_backward")

    compare_device_bench(
        result_csv_path,
        details_csv_path,
        bench_dir,
        device_dir,
        out_path,
        bench_back_dir,
        device_back_dir,
    )


def compare_device_bench(
    result_csv_path,
    details_csv_path,
    bench_dir,
    device_dir,
    out_path,
    bench_grad_dir=None,
    device_grad_dir=None,
):
    Warning_list = []
    compare = Comparator(result_csv_path, details_csv_path, False)
    with FileOpen(result_csv_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
    api_pt_files_bench = os.listdir(bench_dir)
    api_pt_files_device = os.listdir(device_dir)
    api_pt_files_all = list(set(api_pt_files_bench + api_pt_files_device))
    api_pt_files_all = sorted(api_pt_files_all)
    
    # f = open(out_path + "compare_result.txt", 'a', encoding='utf-8')
    errors = []
    errors_forward_info = []
    errors_bacward_info = []
    for i, api_file in enumerate(tqdm.tqdm(api_pt_files_all, **tqdm_params)):
        if not i % dist.get_world_size() == dist.get_rank():
            continue
        bench_out_tensor, device_out_tensor = None, None
        bench_grad_tensor_list, device_grad_tensor_list = None, None
        try:
            print("=" * 100)
            bench_pt_path = os.path.join(bench_dir, api_file)
            device_pt_path = os.path.join(device_dir, api_file)
            if os.path.exists(bench_pt_path) and os.path.exists(device_pt_path):
                print(f"Loading {bench_pt_path} & {device_pt_path}")
                bench_BF16_flag, bench_out_tensor = paddle.load(bench_pt_path)
                device_BF16_flag, device_out_tensor = paddle.load(device_pt_path)
            elif os.path.exists(bench_pt_path) or os.path.exists(device_pt_path):
                msg = f"{api_file} One framework has No output!"
                Warning_list.append(msg)
                print(msg)
                continue
            else:
                msg = f"{api_file} has no output, please refer to run_ut warning log info."
                Warning_list.append(msg)
                print(msg)
                continue

            if bench_grad_dir and device_grad_dir:
                bench_grad_path = os.path.join(bench_grad_dir, api_file)
                device_grad_path = os.path.join(device_grad_dir, api_file)
                if os.path.exists(bench_grad_path) and os.path.exists(device_grad_path):
                    _, bench_grad_tensor_list = paddle.load(bench_grad_path)
                    _, device_grad_tensor_list = paddle.load(device_grad_path)
                    print(f"Loading {bench_grad_path} & {device_grad_path}")
                elif os.path.exists(bench_grad_path) or os.path.exists(
                    device_grad_path
                ):
                    msg = f"{api_file} One framework has No gard output!"
                    Warning_list.append(msg)
                    print(msg)
                else:
                    msg = f"{api_file} has no grad output, please refer to run_ut warning log info."
                    Warning_list.append(msg)
                    print(msg)
            
            error_i = []
            msg = f"{api_file} forward -------------"
            Warning_list.append(msg)
            print(msg)
            compare_result(bench_out_tensor, device_out_tensor, error_i, api_file + " forward")
            errors_forward_info = errors_forward_info + error_i
            
            error_i = []
            msg = f"{api_file} backward -------------"
            Warning_list.append(msg)
            print(msg)
            compare_result(bench_grad_tensor_list, device_grad_tensor_list, error_i, api_file + " backward")
            errors_bacward_info = errors_bacward_info + error_i
        except Exception as err:
            print(err)
    errors_bacward_info.sort(key=lambda x: x[1])
    errors_forward_info.sort(key=lambda x: x[1])
    df = pd.DataFrame(errors_bacward_info, columns=["operator_name", "error<0.001", "bench_data", "device_data", "diff_value", "diff_index"])
    df.to_csv("log/rank" + str(dist.get_rank()) + "_backward_output.csv", index=False)
    df = pd.DataFrame(errors_forward_info, columns=["operator_name", "error<0.001", "bench_data", "device_data", "diff_value", "diff_index"])
    df.to_csv("log/rank" + str(dist.get_rank()) + "_forward_output.csv", index=False)
    
    warning_log_pth = os.path.join(out_path, "./compare_warning.txt")
    File = open(warning_log_pth, "w")
    for item in Warning_list:
        File.write(item + "\n")
    File.close()

def normalize_t(tensor0, tensor1):
    min_val0, min_val1 = paddle.min(tensor0), paddle.min(tensor1)
    max_val0, max_val1 = paddle.max(tensor0), paddle.max(tensor1)
    min_val = min(min_val0, min_val1)
    max_val = max(max_val0, max_val1)
    if len(tensor0) == 1:
        return tensor0 / max_val, tensor1 / max_val
    if min_val == max_val:
        return paddle.ones_like(tensor0), paddle.ones_like(tensor1)
    return (tensor0 - min_val) / (max_val - min_val), (tensor1 - min_val) / (max_val - min_val)

def compare_result(bench_output, device_output, errors, name):
    if isinstance(bench_output, (list, tuple)):
        for b_out_i, n_out_i in zip(bench_output, device_output):
            compare_result(b_out_i, n_out_i, errors, name)
    if isinstance(bench_output, paddle.Tensor):
        bench_output_o = bench_output.reshape([-1,])
        device_output_o = device_output.reshape([-1,])
        bench_output, device_output = normalize_t(bench_output_o, device_output_o)
        # bench_output = paddle.cast(bench_output, "float")
        # device_output = paddle.cast(device_output, "float")
        diff = paddle.cast((bench_output - device_output).abs(), "float")
        num = len(diff)
        diff005 = (diff < 0.05).sum() / num
        diff001 = (diff < 0.01).sum() / num
        diff0005 = (diff < 0.005).sum() / num
        diff0001 = (diff < 0.001).sum() / num
        diff00005 = (diff < 0.0005).sum() / num
        if diff0001 < 1 or len(bench_output) == 1:
            diff_value, diff_index = paddle.topk(diff, k=min(10, num))
            error_info = diff0001.numpy()
            bench_n = paddle.cast(bench_output_o[diff_index], "float").numpy().tolist()
            device_n = paddle.cast(device_output_o[diff_index], "float").numpy().tolist()
            diff_index_n = diff_index.numpy().tolist()
            diff_value_n = diff_value.numpy().tolist()
            errors.append((name, error_info, str(bench_n), str(device_n), str(diff_value_n), str(diff_index_n)))
            print("diff is too large---------------------------- erorr Erorr ERORR----------------------------")
            print("bench_output----------")
            print(bench_output_o[diff_index])
            print("device_output---------")
            print(device_output_o[diff_index])
        print("diff < 0.05:   ", diff005.numpy())
        print("diff < 0.01:   ", diff001.numpy())
        print("diff < 0.005:  ", diff0005.numpy())
        print("diff < 0.001:  ", diff0001.numpy())
        print("diff < 0.0005: ", diff00005.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _compare_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    compare_command(args)
