import argparse
import os
import paddle
import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        ops = yaml.safe_load(f)
    return ops

def write_yaml(file_path, ops):
    with open(file_path, 'w') as f:
        yaml.dump(ops, f, allow_unicode=True, default_flow_style=False)

def update_yaml(ops, prefix_list):
    new_ops = []
    for prefix in prefix_list:
        tmp = dir(eval(prefix))
        # filter out "__xxx__" and inplace elements
        tmp = [prefix + "." + op for op in tmp if not op.startswith("__") and not op.endswith("_")]
        new_ops.extend(tmp)

    target_ops = ops["target_op"]
    target_ops = [op for op in target_ops if not op.startswith(prefix)]

    merged_ops = target_ops + new_ops
    ops["target_op"] = merged_ops

def arg_parser(parser):
    parser.add_argument(
        "-yaml",
        "--yaml",
        dest="yaml_path",
        default="../api_tracer/configs/op_target.yaml",
        type=str,
        help="op_target yaml file path",
        required=False,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg_parser(parser)
    cfg = parser.parse_args()
    yaml_path = cfg.yaml_path
    assert os.path.exists(yaml_path), "yaml file not exist"
    prefix_list = ["paddle._C_ops", "paddle.nn.functional"]

    ops = read_yaml(yaml_path)
    update_yaml(ops, prefix_list)
    write_yaml(yaml_path, ops)
