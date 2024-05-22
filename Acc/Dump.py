import json
import os
from . import config
import paddle

cfg = config.cfg


def write_pt(file_path, tensor):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(
            f"File {file_path} already exists, tool has overwritten it automatically."
        )
    paddle.save(tensor, file_path)
    full_path = os.path.realpath(file_path)
    return full_path


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
    with open(json_pth, mode="a+") as f:
        json.dump(data, f, indent=2)


class Dump:
    def __init__(self, mode="real_data"):
        self.api_info = {}
        self.data_route = cfg.dump_root_path
        self.mode = mode
        self.rank = None
        self.dump_api_dict = None

    """
        Dump tensor object to disk.
        return: disk route
    """

    def dump_real_data(self, api_args, tensor, rank):
        self.rank = rank
        directory = os.path.join(self.data_route, f"rank{rank}_step{cfg.global_step}")
        file_path = os.path.join(directory, f"{api_args}.pt")
        create_directory(directory)
        pt_path = write_pt(file_path, tensor)
        return pt_path

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

        create_directory(directory)
        if self.rank is not None:
            write_json(directory, self.dump_api_dict, rank=self.rank, mode="forward")
        else:
            write_json(directory, self.dump_api_dict, rank=None, mode="forward")


dump_util = Dump()
