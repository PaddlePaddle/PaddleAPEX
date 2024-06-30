# 进行比对及结果展示
import os
import csv
import argparse
import sys
import time
import paddle
import tqdm

from compare_utils.compare import Comparator
from compare_utils.compare_dependency import print_info_log, FileOpen

current_time = time.strftime("%Y%m%d%H%M%S")

RESULT_FILE_NAME = "accuracy_checking_result_" + current_time + ".csv"
DETAILS_FILE_NAME = "accuracy_checking_details_" + current_time + ".csv"

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
        "-gpu",
        "--benchmark",
        dest="bench_dir",
        type=str,
        help="The executed output api tensor path directory on GPU",
        required=True,
    )
    parser.add_argument(
        "-npu",
        "--device",
        dest="device_dir",
        type=str,
        help="The executed output api tensor path directory on NPU",
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
    gpu_back_dir = os.path.join(args.bench_dir, "../output_backward")
    npu_back_dir = os.path.join(args.device_dir, "../output_backward")

    compare_npu_gpu(
        result_csv_path,
        details_csv_path,
        args.bench_dir,
        args.device_dir,
        out_path,
        gpu_back_dir,
        npu_back_dir,
    )


def compare_npu_gpu(
    result_csv_path,
    details_csv_path,
    bench_dir,
    device_dir,
    out_path,
    gpu_grad_dir=None,
    npu_grad_dir=None,
):
    Warning_list = []
    compare = Comparator(result_csv_path, details_csv_path, False)
    with FileOpen(result_csv_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
    api_pt_files_gpu = os.listdir(bench_dir)
    api_pt_files_npu = os.listdir(device_dir)
    api_pt_files_all = list(set(api_pt_files_gpu + api_pt_files_npu))
    api_pt_files_all = sorted(api_pt_files_all)

    for i, api_file in enumerate(tqdm.tqdm(api_pt_files_all, **tqdm_params)):
        try:
            print("=" * 100)
            bench_pt_path = os.path.join(bench_dir, api_file)
            device_pt_path = os.path.join(device_dir, api_file)
            if os.path.exists(bench_pt_path) and os.path.exists(device_pt_path):
                print(f"Loading {bench_pt_path} & {device_pt_path}")
                bench_out_tensor = paddle.load(bench_pt_path)
                device_out_tensor = paddle.load(device_pt_path)
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

            bench_grad_tensor_list, device_grad_tensor_list = None, None
            if gpu_grad_dir and npu_grad_dir:
                bench_grad_path = os.path.join(gpu_grad_dir, api_file)
                device_grad_path = os.path.join(npu_grad_dir, api_file)
                if os.path.exists(bench_grad_path) and os.path.exists(device_grad_path):
                    bench_grad_tensor_list = paddle.load(bench_grad_path)
                    device_grad_tensor_list = paddle.load(device_grad_path)
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

            compare.compare_output(
                api_file,
                bench_out_tensor,
                device_out_tensor,
                bench_grad_tensor_list,
                device_grad_tensor_list,
            )
        except Exception as err:
            print(err)
    warning_log_pth = os.path.join(out_path, "./compare_warning.txt")
    File = open(warning_log_pth, "w")
    for item in Warning_list:
        File.write(item + "\n")
    File.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _compare_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    compare_command(args)
