# 进行比对及结果展示
import os
import csv
import argparse
import sys
import time
import paddle
import tqdm

from compare.compare import Comparator
from compare.compare_dependency import print_info_log, FileOpen, seed_all

seed_all()

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
        "--input_path1",
        dest="gpu_data_dir",
        type=str,
        help="The executed output api tensor path directory on GPU",
        required=True,
    )
    parser.add_argument(
        "-npu",
        "--input_path2",
        dest="npu_data_dir",
        type=str,
        help="The executed output api tensor path directory on NPU",
        required=True,
    )
    parser.add_argument(
        "-gpu_back",
        "--input_backward_path1",
        dest="gpu_back_dir",
        default="",
        type=str,
        help="The api param tool backward result directory on GPU",
        required=False,
    )
    parser.add_argument(
        "-npu_back",
        "--input_backward_path2",
        dest="npu_back_dir",
        default="",
        type=str,
        help="The api param tool backward result directory on NPU",
        required=False,
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
    result_csv_path = os.path.join(out_path, RESULT_FILE_NAME)
    details_csv_path = os.path.join(out_path, DETAILS_FILE_NAME)
    print_info_log(f"Compare task result will be saved in {result_csv_path}")
    print_info_log(f"Compare task details will be saved in {details_csv_path}")
    compare_npu_gpu(
        result_csv_path,
        details_csv_path,
        args.gpu_data_dir,
        args.npu_data_dir,
        args.gpu_back_dir,
        args.npu_back_dir,
    )


def compare_npu_gpu(
    result_csv_path,
    details_csv_path,
    gpu_data_dir,
    npu_data_dir,
    gpu_grad_dir=None,
    npu_grad_dir=None,
):
    compare = Comparator(result_csv_path, details_csv_path, False)
    with FileOpen(result_csv_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
    api_pt_files_gpu = os.listdir(gpu_data_dir)
    api_pt_files_npu = os.listdir(npu_data_dir)
    api_pt_files_all = list(set(api_pt_files_gpu + api_pt_files_npu))
    api_pt_files_all = sorted(api_pt_files_all)

    for i, api_file in enumerate(tqdm.tqdm(api_pt_files_all, **tqdm_params)):
        try:
            name = api_file.split("*")
            print(name)
            gpu_pt_path = os.path.join(gpu_data_dir, api_file)
            npu_pt_path = os.path.join(npu_data_dir, api_file)
            print("Loading:")
            print(gpu_pt_path)
            print(npu_pt_path)
            gpu_out_tensor = paddle.load(gpu_pt_path)
            npu_out_tensor = paddle.load(npu_pt_path)
            print("gpu Tensor: ", gpu_out_tensor.dtype.name)
            print("npu Tensor: ", npu_out_tensor.dtype.name)

            gpu_grad_tensor_list, npu_grad_tensor_list = None, None
            if gpu_grad_dir and npu_grad_dir:
                gpu_grad_path = os.path.join(gpu_grad_dir, api_file)
                npu_grad_path = os.path.join(npu_grad_dir, api_file)
                if os.path.exists(gpu_grad_path):
                    gpu_grad_tensor_list = paddle.load(gpu_grad_path)
                    npu_grad_tensor_list = paddle.load(npu_grad_path)
                    print_info(gpu_grad_tensor_list, npu_grad_tensor_list)
                else:
                    print(
                        f"{api_file} Doesn't exist! Please check BP_LIST in run_dualback_ut.py"
                    )

            compare.compare_output(
                api_file,
                gpu_out_tensor,
                npu_out_tensor,
                gpu_grad_tensor_list,
                npu_grad_tensor_list,
            )
        except Exception as err:
            print(err)


def print_info(tensor_list1, tensor_list2):
    for item1, item2 in zip(tensor_list1, tensor_list2):
        if isinstance(item1, paddle.Tensor):
            print("Load device1 grad Tensor: ", item1.dtype.name)
        if isinstance(item2, paddle.Tensor):
            print("Load device2 grad Tensor: ", item2.dtype.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _compare_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    paddle.set_device("cpu")
    compare_command(args)
