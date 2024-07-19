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
import re
import argparse
import sys

from compare_utils.compare_dependency import print_info_log

RESULT_FILE_NAME = "prof_checking_result" + ".csv"
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
        bench_profile_log = os.path.join(args.bench_dir, "./profile_analyze.log")
        device_profile_log = os.path.join(args.device_dir, "./profile_analyze.log")
    except FileNotFoundError:
        print_info_log("The log file is not found.")
    compare_device_bench(
        result_csv_path,
        bench_profile_log,
        device_profile_log,
    )


def analyze_log(raw_data):
    res_dict = {}
    for item in raw_data:
        single_op_dict = {}
        item = item.replace('\n', '')
        data_list = item.split("\t")
        single_op_dict["dtype"] = data_list[2]
        single_op_dict["input shape"] = data_list[4]
        single_op_dict["output shape"] = data_list[6]
        single_op_dict["direction"] = data_list[7]
        single_op_dict["Time us"] = data_list[8]
        op_name = data_list[0] + "*" + data_list[2]
        res_dict[op_name] = single_op_dict
    return res_dict

def get_cmp_result_prof(value1, value2):
    value1 = float(value1)
    value2 = float(value2)
    return str(value2 / value1)


def compare_device_bench(
    result_csv_path,
    bench_profile_log,
    device_profile_log,
):
    ensemble_data = []
    with open(bench_profile_log, "r") as prof_f1:
        prof_lines = prof_f1.readlines()
        prof_f1.close()
    prof_dict1 = analyze_log(prof_lines)
    with open(device_profile_log, "r") as prof_f2:
        prof_lines = prof_f2.readlines()
        prof_f2.close()
    prof_dict2 = analyze_log(prof_lines)
    union_keys = set(prof_dict1.keys()) | set(prof_dict2.keys())

    for key in union_keys:
        temp_dict = {}
        temp_dict["API Name"] = key
        temp_dict["dtype"] = "None"
        temp_dict["input shape"] = "None"
        temp_dict["output shape"] = "None"
        temp_dict["direction"] = "None"
        temp_dict["Bench Time(us)"] = "None"
        temp_dict["Device Time(us)"] = "None"
        temp_dict["Device/Bench Time Ratio"] = "None"

        if key in prof_dict1.keys():
            temp_dict["API Name"] = key
            temp_dict["dtype"] = prof_dict1[key]["dtype"]
            temp_dict["input shape"] = prof_dict1[key]["input shape"]
            temp_dict["output shape"] = prof_dict1[key]["output shape"]
            temp_dict["direction"] = prof_dict1[key]["direction"]
            temp_dict["Bench Time(us)"] = prof_dict1[key]["Time us"]
        if key in prof_dict2.keys():
            temp_dict["Device Time(us)"] = prof_dict2[key]["Time us"]
            if key in prof_dict1.keys():
                temp_dict["Device/Bench Time Ratio"] = get_cmp_result_prof(prof_dict1[key]["Time us"], prof_dict2[key]["Time us"])
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
