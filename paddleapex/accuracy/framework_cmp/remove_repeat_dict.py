'''
    This script is used to remove the repeated api input cases in a JSON file.
'''
import json

def remove_fields(dictionary):
    dictionary_copy = dictionary.copy()
    for key, value in dictionary_copy.items():
        if isinstance(value, dict):
            remove_fields(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    remove_fields(item)
        elif isinstance(value, float):
            if "Max" in dictionary:
                del dictionary["Max"]
            if "Max_origin" in dictionary:
                del dictionary["Max_origin"]
            if "Min" in dictionary:
                del dictionary["Min"]
            if "Min_origin" in dictionary:
                del dictionary["Min_origin"]
        elif isinstance(value, bool):
            if "stop_gradient" in dictionary:
                del dictionary["stop_gradient"]
    return dictionary

# 从JSON文件中读取字典数据
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 将字典数据保存到JSON文件中
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

input_file_path = "ernie.json"  # 替换为您的输入JSON文件路径
output_file_path = "debug.json"  # 替换为您的输出JSON文件路径

data = read_json(input_file_path)

# 进行字典处理
filtered_data = {}

OP_LIST = []

for key,values in data.items():
    name = key.split("*")[0] + "*" + key.split("*")[1]
###########
    name = name+ "*0"
    filtered_data[name] = {
        'args': values['args'],
        'kwargs': values['kwargs']
    }
save_json(filtered_data, output_file_path)