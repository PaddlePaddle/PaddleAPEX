"""
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
"""
"""
    This script is used to transfer the json file from paddle format to torch format.
"""

import json
import inspect
import argparse
import paddle  # noqa

parser = argparse.ArgumentParser()
parser.add_argument(
    "-mapping",
    dest="mapping_json",
    default="./api_mapping.json",
    type=str,
    help="Dump json file path",
    required=False,
)
parser.add_argument(
    "-json_path",
    "--json",
    dest="json_path",
    default="./sample_dump.json",
    type=str,
    help="",
    required=False,
)


if __name__ == "__main__":
    cfg = parser.parse_args()
    mapping_json = cfg.mapping_json
    map_F = open(mapping_json, "r")
    mapping = json.loads(map_F.read())

    INPLACE_OP = mapping["inplace_api"]
    api_mapping = mapping["mapping"]
    Warning_list = []
    json_path = cfg.json_path
    W = open("./Json_transfer_warning.log", "a")
    F = open(json_path, "r")
    Paddle_F = open(json_path[:-5] + "_paddle.json", "w")
    Torch_F = open(json_path[:-5] + "_torch.json", "w")
    content = json.loads(F.read())
    F.close()

    res_paddle_dict = {}
    res_torch_dict = {}
    number = 0
    for item in content:
        # item: dumped api info.
        func_stack = item.split("*")[0]
        func = eval(func_stack)
        try:
            # full args_list of paddle api
            args_name_list = inspect.getfullargspec(func).args
            inplace_flag = False
        except Exception:
            inplace_flag = True
            msg = f"Cannot obtain {item} args names!"
            W.write(msg + "\n")
            print(msg)

        api_info = item.split("*")[0]
        if api_info in INPLACE_OP:
            inplace_flag = True

        # dumped api args_list
        args_list = content[item]["args"]
        single_paddle_op = {}
        single_paddle_op["dout_list"] = content[item]["dout_list"]
        single_paddle_op["kwargs"] = {}
        if inplace_flag:
            single_paddle_op["args"] = args_list
        else:
            # Paddle kwargs analyze
            for idx, variable in enumerate(args_list):
                single_paddle_op["kwargs"].update({args_name_list[idx]: variable})
        kwargs_dict = content[item]["kwargs"]
        for k, v in kwargs_dict.items():
            single_paddle_op["kwargs"].update({k: v})
        res_paddle_dict[item] = single_paddle_op

        op_name = item.split("*")[0]
        try:
            torch_call_stack = (
                api_mapping[op_name]["torch_api"] + "*" + item.split("*")[1]
            )
        except Exception:
            number += 1
            msg = f"Paddle api {op_name} has no matched api in torch."
            Warning_list.append(msg)

            torch_call_stack = f"unmatched_op*{number}"
            single_torch_op = []
            single_torch_op = single_paddle_op
            single_torch_op["origin_paddle_op"] = item
            res_torch_dict[torch_call_stack] = single_torch_op
            continue

        # Torch kwargs analyze
        single_torch_op = {}
        single_torch_op["args"] = []
        single_torch_op["kwargs"] = {}
        single_torch_op["origin_paddle_op"] = item
        single_torch_op["dout_list"] = content[item]["dout_list"]

        if inplace_flag:
            single_torch_op["args"] = args_list
            single_torch_op["inplace"] = True
        else:
            single_torch_op["inplace"] = False
        try:
            kwargs_change_dict = api_mapping[op_name]["kwargs_change"]
        except Exception:
            kwargs_change_dict = {}

        for key, variable in single_paddle_op["kwargs"].items():
            try:
                torch_args_list = api_mapping[op_name]["torch_args_list"]
            except Exception:
                torch_args_list = None
            if torch_args_list is not None:
                if kwargs_change_dict is not None:
                    if key in torch_args_list:
                        single_torch_op["kwargs"].update({key: variable})
                    elif key in kwargs_change_dict:
                        single_torch_op["kwargs"].update(
                            {kwargs_change_dict[key]: variable}
                        )
                    elif key in api_mapping[op_name]["unsupport_args"]:
                        # If torch has unmatched args, please remove it from torch args list, and append to unsupport list!
                        msg = f"{op_name} {key} is not supported in torch, It could cause error in comparision!"
                        print(msg)
                        Warning_list.append(msg)
                    else:
                        msg = f"{op_name} Cannot idetify key word: {key}."
                        print(msg)
                        Warning_list.append(msg)
        res_torch_dict[torch_call_stack] = single_torch_op
    json.dump(res_paddle_dict, Paddle_F, indent=2)
    json.dump(res_torch_dict, Torch_F, indent=2)
