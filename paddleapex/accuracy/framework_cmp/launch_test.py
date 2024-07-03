import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-json",
    "--json",
    dest="json_path",
    default="./sample_dump.json",
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
parser.add_argument(
    "-op",
    dest="debug_op",
    default="",
    type=str,
    help="",
    required=False,
)
cfg = parser.parse_args()

cwd = __file__.rsplit("/", 1)[0]
out_dir = "./auto_cmp_repo/"
enforce_dtype = cfg.multi_dtype_ut
json_prefix = cfg.json_path[:-5]
out_dir_paddle = out_dir + "paddle"
out_dir_torch = out_dir + "torch"


paddle_script = os.path.join(cwd, "run_paddle.py")
torch_script = os.path.join(cwd, "run_torch.py")
transfer_script = os.path.join(cwd, "json_transfer.py")
mapping_script = os.path.join(cwd, "api_mapping.json")
cmp_script = os.path.join(cwd, "../direct_cmp.py")

print("Current workspace directory:", cwd)
print(f"Paddle script: {paddle_script}")
print(f"Torch script: {torch_script}")
print(f"Mapping script: {mapping_script}")
print(f"Json transfer script: {transfer_script}")
# Json transfer
json_transfer_cmd = [
    "python",
    transfer_script,
    "-mapping",
    mapping_script,
    "-json_path",
    cfg.json_path[:-5] + ".json",
]
subprocess.run(json_transfer_cmd)

# paddle case
command1 = [
    "python",
    paddle_script,
    "-json",
    json_prefix + "_paddle.json",
    "-out",
    out_dir_paddle,
    "-enforce",
    enforce_dtype,
    "-op",
    cfg.debug_op,
]
subprocess.run(command1)


# torch case
command2 = [
    "python",
    torch_script,
    "-json",
    json_prefix + "_torch.json",
    "-out",
    out_dir_torch,
    "-enforce",
    enforce_dtype,
    "-op",
    cfg.debug_op,
]
subprocess.run(command2)

for item in ["BF16", "FP32", "FP16"]:
    out_dir_paddle_forward = os.path.join(out_dir_paddle, item)
    out_dir_torch_forward = os.path.join(out_dir_torch, item)
    cmp_command = [
        "python",
        cmp_script,
        "-gpu",
        out_dir_paddle_forward,
        "-npu",
        out_dir_torch_forward,
        "-o",
        out_dir,
    ]
    subprocess.run(cmp_command)
