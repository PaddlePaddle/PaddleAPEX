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


def write_json(file_path, data, rank=None, mode="forward"):
    if rank is not None:
        json_pth = os.path.join(file_path, mode + "_rank" + str(rank) + ".json")
    else:
        json_pth = os.path.join(file_path, mode + ".json")
    if os.path.exists(json_pth):
        os.remove(json_pth)
        print(f"File {json_pth} already exists, tool has overwritten it automatically.")
    with open(json_pth, mode="w") as f:
        json.dump(data, f, indent=2)


class Dump:
    def __init__(self, mode="real_data", Async_save=cfg.Async_dump):
        self.api_info = {}
        self.data_route = cfg.dump_root_path
        self.mode = mode
        self.rank = None
        self.dump_api_dict = None
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

    def update_api_dict(self, api_info_dict, rank):
        self.rank = rank
        if self.dump_api_dict is None:
            self.dump_api_dict = api_info_dict
        else:
            self.dump_api_dict.update(api_info_dict)

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
        create_directory(directory)
        if self.rank is not None:
            write_json(directory, self.dump_api_dict, rank=self.rank, mode="forward")
        else:
            write_json(directory, self.dump_api_dict, rank=None, mode="forward")


dump_util = Dump()
