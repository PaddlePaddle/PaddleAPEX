
import subprocess

out_dir = f"./framwork_compare/"
out_dir_paddle = out_dir + "/paddle"
command1 = ["python", "run_paddle.py", "--forward", "./paddle.json", "-o", out_dir_paddle]
subprocess.run(command1)


out_dir_torch = out_dir + "/torch"
command2 = ["python", "run_torch.py", "--forward", "./torch.json", "-o", out_dir_torch]
subprocess.run(command2)

out_dir_paddle_forward = out_dir_paddle + "/gpu_output"
out_dir_paddle_backward = out_dir_paddle + "/gpu_output_backward"

out_dir_torch_forward = out_dir_torch + "/torch_output"
out_dir_torch_backward = out_dir_torch + "/torch_output_backward"

cmp_command = ["python", "../compare.py", "-gpu", out_dir_paddle_forward, "-npu", out_dir_torch_forward, "-gpu_back", out_dir_paddle_backward, "-npu_back", out_dir_torch_backward, "-o", out_dir]
subprocess.run(cmp_command)

