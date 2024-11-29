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
import json
import os
from paddleapex.api_tracer.config import cfg
from paddleapex.utils import ThreadPool, save_tensor


def create_directory(data_route):
    try:
        os.makedirs(data_route, exist_ok=True)
    except OSError as ex:
        print("In create_directory: for dump_path:{}, {}".format(data_route, str(ex)))

def write_json(file_path, data, rank=None, mode="forward", split_type="all"):
    if rank is not None:
        json_pth = os.path.join(file_path, mode + "_rank" + str(rank) + "_" + split_type + ".json")
    else:
        json_pth = os.path.join(file_path, mode + "_" + split_type + ".json")
    if os.path.exists(json_pth):
        os.remove(json_pth)
        print(f"File {json_pth} already exists, tool has overwritten it automatically.")
    with open(json_pth, mode="w") as f:
        json.dump(data, f, indent=2)

def get_unique_api_dict(dump_api_dict):
    if dump_api_dict == None:
        return {}

    SORT_KEY_SEPARATOR = "*"
    sorted_info = dict(sorted(dump_api_dict.items(), key=lambda item: item[0]))

    result_dict = {}
    current_operation = ""
    unique_values = set()
    unique_count = 0

    for key, value in sorted_info.items():
        operation = key.split(SORT_KEY_SEPARATOR)[0]
        value_str = str(value)

        if operation != current_operation:
            current_operation = operation
            unique_values.clear()
            unique_count = 0

        if value_str not in unique_values:
            unique_values.add(value_str)
            result_dict[f"{current_operation}{SORT_KEY_SEPARATOR}{unique_count}"] = value
            unique_count += 1

    return result_dict

class Dump:
    def __init__(self, mode="real_data", Async_save=cfg.Async_dump):
        self.api_info = {}
        self.data_route = cfg.dump_root_path
        self.mode = mode
        self.rank = None
        self.dump_api_dict = None
        self.dump_api_dict_half = None
        self.dump_api_dict_distributed = None
        self.dump_api_dict_other = None
        self.Async_save = Async_save

        if self.Async_save:
            self.pool = ThreadPool()
        else:
            pass

    """
        Dump tensor object to disk.
        return: disk route
    """

    def dump_real_data(self, api_args, tensor, rank):
        self.rank = rank
        directory = os.path.join(self.data_route, f"rank{rank}_step{cfg.global_step}")
        file_path = os.path.join(directory, f"{api_args}.pt")
        create_directory(directory)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(
                f"File {file_path} already exists, tool has overwritten it automatically."
            )
        if self.Async_save:
            remote_repo = os.path.join(
                cfg.remote_path, f"rank{rank}_step{cfg.global_step}"
            )
            create_directory(remote_repo)
            self.pool.safe_parellel_save(tensor, file_path, remote_repo)
        else:
            save_tensor(tensor, file_path)
        return f"{api_args}.pt"

    """
        Get Api_info dict, update self.dump_api_dict
    """

    def update_api_dict(self, api_info_dict, rank, is_half_precision = False, is_distributed = False):
        self.rank = rank
        if self.dump_api_dict is None:
            self.dump_api_dict = api_info_dict.copy()
        else:
            self.dump_api_dict.update(api_info_dict)
        
        if cfg.split_dump:
            if is_distributed:
                if self.dump_api_dict_distributed is None:
                    self.dump_api_dict_distributed = api_info_dict.copy()
                else:
                    self.dump_api_dict_distributed.update(api_info_dict)
            if is_half_precision:
                if self.dump_api_dict_half is None:
                    self.dump_api_dict_half = api_info_dict.copy()
                else:
                    self.dump_api_dict_half.update(api_info_dict)
            else:
                if self.dump_api_dict_other is None:
                    self.dump_api_dict_other = api_info_dict.copy()
                else:
                    self.dump_api_dict_other.update(api_info_dict)

    def dump(self):
        if self.rank is not None:
            directory = os.path.join(
                self.data_route, f"rank{self.rank}_step{cfg.global_step}"
            )
        else:
            directory = self.data_route
        if self.dump_api_dict is None:
            print(
                "Dump api dict is empty, check if you have correctly inserted marks into scripts"
            )
            print("Especially in pipeline parallel mode!")
        if cfg.dump_unique:
            self.dump_api_dict = get_unique_api_dict(self.dump_api_dict)
            if cfg.split_dump:
                self.dump_api_dict_half = get_unique_api_dict(self.dump_api_dict_half)
                self.dump_api_dict_other = get_unique_api_dict(self.dump_api_dict_other)
        create_directory(directory)
        if self.rank is not None:
            write_json(directory, self.dump_api_dict, rank=self.rank, mode="forward", split_type="all")
            if cfg.split_dump:
                write_json(directory, self.dump_api_dict_half, rank=self.rank, mode="forward", split_type="half")
                write_json(directory, self.dump_api_dict_distributed, rank=self.rank, mode="forward", split_type="distributed")
                write_json(directory, self.dump_api_dict_other, rank=self.rank, mode="forward", split_type="other")
        else:
            write_json(directory, self.dump_api_dict, rank=None, mode="forward", split_type="all")
            if cfg.split_dump:
                write_json(directory, self.dump_api_dict_half, rank=None, mode="forward", split_type="half")
                write_json(directory, self.dump_api_dict_distributed, rank=None, mode="forward", split_type="distributed")
                write_json(directory, self.dump_api_dict_other, rank=None, mode="forward", split_type="other")


dump_util = Dump()
