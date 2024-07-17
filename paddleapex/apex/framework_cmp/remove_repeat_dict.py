# 原地删除dump json中相同dtype、相同shape的api信息。
import json
import copy
import argparse


# 传入api信息，删除特定字段，用于去重。 传入前必须先deepcopy
def remove(arg_in):
    if isinstance(arg_in, (list, tuple)):
        res = []
        for item in arg_in:
            ret_value = remove(item)
            res.append(ret_value)
    elif isinstance(arg_in, dict):
        if "Max" in arg_in.keys():
            del arg_in["Max"]
        if "Max_origin" in arg_in.keys():
            del arg_in["Max_origin"]
        if "Min" in arg_in.keys():
            del arg_in["Min"]
        if "Min_origin" in arg_in.keys():
            del arg_in["Min_origin"]
        if "stop_gradient" in arg_in.keys():
            del arg_in["stop_gradient"]
        for k, v in arg_in.items():
            remove(v)
        return arg_in


# 从JSON文件中读取字典数据
def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


# 将字典数据保存到JSON文件中
def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-json",
    dest="input_file_path",
    type=str,
    help="Dump json file path",
    required=True,
)

cfg = parser.parse_args()

input_file_path = cfg.input_file_path
output_file_path = input_file_path[:-5] + "_unique.json"
data = read_json(input_file_path)

OP_LIST = []
for key, values in data.items():
    name = key.split("*")[0]
    OP_LIST.append(name)

OP_LIST = set(OP_LIST)
OP_LIST = list(OP_LIST)
OP_set = {}
# 遍历所有op，形成一个唯一op_name的列表，每次循环会拿到该op的所有case
# case包括不同dtype、不同shape、不同超参。
for item in OP_LIST:
    op_name = item
    count = 0
    op_args = []
    for key in data:  # 根据op_name，从dumpjson中寻找相同op_name的信息
        if op_name == key.split("*")[0]:
            raw_data_copy = copy.deepcopy(data[key])
            remove(raw_data_copy)
            unique_flag = True
            for dict_i in op_args:
                dict_copy = copy.deepcopy(dict_i)
                remove(dict_copy)
                # input()
                if raw_data_copy == dict_copy:
                    unique_flag = False
                    break

            if unique_flag:
                op_args.append(data[key])

    count = len(op_args)
    for i in range(count):
        OP_set.update({f"{op_name}*{i}": op_args[i]})

save_json(OP_set, output_file_path)
