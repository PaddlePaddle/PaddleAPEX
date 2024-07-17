# 进行比对及结果展示
import os
import csv
import re
import argparse
import sys
import math

from compare_utils.compare_dependency import print_info_log

RESULT_FILE_NAME = "memory_checking_result" + ".csv"
TIME_RATIO = 0.95  # BENCH_TIME / DEVICE_TIME >= 0.95 as standard.
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
    print_info_log(f"Compare task result will be saved in {result_csv_path}")
    try:
        bench_mem_log = os.path.join(args.bench_dir, "./memory_analyze.log")
        device_mem_log = os.path.join(args.device_dir, "./memory_analyze.log")
    except FileNotFoundError:
        print_info_log("The log file is not found.")
    compare_device_bench(
        result_csv_path,
        bench_mem_log,
        device_mem_log,
    )


def analyze_log(raw_data):
    res_dict = {}
    pattern = r"^(.*?)\s*:\t(.*?)\n$"
    for item in raw_data:
        match = re.match(pattern, item)
        if match:
            api_name = match.group(1)
            data = match.group(2)
            res_dict[api_name] = data
        else:
            print("The format of log is not correct.")
    return res_dict


def get_cmp_result_mem(value1, value2):
    return str(int(value2) - int(value1))


def compare_device_bench(
    result_csv_path,
    bench_mem_log,
    device_mem_log,
):
    ensemble_data = []
    with open(bench_mem_log, "r") as mem_f1:
        mem_lines = mem_f1.readlines()
        mem_f1.close()
    mem_dict1 = analyze_log(mem_lines)
    with open(device_mem_log, "r") as mem_f2:
        mem_lines = mem_f2.readlines()
        mem_f2.close()
    mem_dict2 = analyze_log(mem_lines)
    print(mem_dict1)
    union_keys = set(mem_dict1.keys()) | set(mem_dict2.keys())

    for key in union_keys:
        temp_dict = {}
        if key in mem_dict1.keys():
            temp_dict["API Name"] = key
            temp_dict["Bench Memory Usage (B)"] = mem_dict1[key]
        if key in mem_dict2.keys():
            temp_dict["Device Memory Usage (B)"] = mem_dict2[key]
            if key in mem_dict1.keys():
                temp_dict["Memory Difference"] = abs(float(mem_dict1[key])-float(mem_dict2[key]))
        else:
            temp_dict["Device Memory Usage (B)"] = ""
            temp_dict["Memory Difference"] = ""
        ensemble_data.append(temp_dict)

    with open(result_csv_path, "w", newline="") as file:
        fieldnames = ensemble_data[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in ensemble_data:
            writer.writerow(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _compare_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    compare_command(args)
