import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-json",
    "--json",
    dest="json_path",
    default="",
    type=str,
    help="Dump json file path",
    required=True,
)
parser.add_argument(
    "-enforce",
    "--dtype",
    dest="multi_dtype_ut",
    default="FP32,FP16,BF16",
    type=str,
    help="",
    required=False,
)
cfg = parser.parse_args()

out_dir = "./auto_cmp_repo/"
enforce_dtype = cfg.multi_dtype_ut
json_prefix = cfg.json_path[:-5]
out_dir_paddle = out_dir + "paddle"
out_dir_torch = out_dir + "torch"


# Json transfer
json_transfer_cmd = [
    "python",
    "json_transfer.py",
    "-mapping",
    "./api_mapping.json",
    "-json_path",
    json_prefix + ".json",
]
subprocess.run(json_transfer_cmd)

# paddle case
command1 = [
    "python",
    "run_paddle.py",
    "-json",
    json_prefix + "_paddle.json",
    "-o",
    out_dir_paddle,
    "-enforce",
    enforce_dtype,
]
subprocess.run(command1)


# torch case
command2 = [
    "python",
    "run_torch.py",
    "-json",
    json_prefix + "_torch.json",
    "-o",
    out_dir_torch,
    "-enforce",
    enforce_dtype,
]
subprocess.run(command2)

for item in ["BF16", "FP32", "FP16"]:
    out_dir_paddle_forward = os.path.join(out_dir_paddle, item)
    out_dir_torch_forward = os.path.join(out_dir_torch, item)
    cmp_command = [
        "python",
        "../direct_cmp.py",
        "-gpu",
        out_dir_paddle_forward,
        "-npu",
        out_dir_torch_forward,
        "-o",
        out_dir,
    ]
    subprocess.run(cmp_command)
