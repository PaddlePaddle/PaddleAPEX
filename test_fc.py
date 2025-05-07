import paddle
import paddle.nn as nn
import json
import numpy as np
import os
import re
import math

from tools import *

try:
    from paddle_xpu.layers.nn import Linear
except ImportError:
    from paddle.nn import Linear
#from paddle.nn import Linear

#log_file_path = 'workerlog.0.fc_fa_mean_max'
log_file_path = 'fc.log'

pattern = re.compile(r".*?cases(.*)")
mean_var = r"\[mean\](-?\d+\.\d+), \[max\](\d+\.\d+)"


input_file = 'FC_INPUT'
xpu_file = 'XPU'
cal_out_file = xpu_file + '/output'
cal_out_back_file = xpu_file + '/output_backward'
base_out_file = 'BASE/output'
base_out_back_file = 'BASE/output_backward'

createDir(input_file)
createDir(xpu_file)

CREATE_DATA = False
DEVICE = 'XPU'
#DEVICE = 'GPU'
base_type = paddle.float32
dtype = paddle.bfloat16
#calculate_type = paddle.bfloat16
calculate_type = base_type


if calculate_type == base_type:
    createDir(base_out_file)
    out_file = base_out_file
    createDir(base_out_back_file)
    out_back_file = base_out_back_file
else:
    createDir(cal_out_file)
    out_file = cal_out_file
    createDir(cal_out_back_file)
    out_back_file = cal_out_back_file

paddle.set_default_dtype(calculate_type)


# 初始化列表
list_a = []
list_b = []
list_c = []
list_d = []
alpha_beta = []
a_mean = []
b_mean = []
d_mean = []
a_max = []
b_max = []
d_max = []
n = 0
try:
    with open(log_file_path, 'r') as file:
        log_lines = file.readlines()

    for i in range(len(log_lines)):
        print("i", i)
        line = log_lines[i]
        if "kXPU3" not in log_lines[i]:
            continue
        if "float16" not in log_lines[i]:
            continue
        #if "fc_fusion" in log_lines[i]:
        #    run_mode = log_lines[i + 3]
        #    if "desc.run_mode" in run_mode:
        #        mean_match = re.search(mean_var, run_mode)
        #        if mean_match:
        #            if float(mean_match.group(1)) == 1:
        #                continue
        #        else:
        #            print('------------------error----------------')
        #else:
        #    continue

        #a_line = log_lines[i + 1]
        #match = re.search(mean_var, a_line)
        #if match:
        #    a_mean.append(float(match.group(1)))
        #    a_max.append(float(match.group(2)))
        #b_line = log_lines[i + 2]
        #match = re.search(mean_var, b_line)
        #if match:
        #    b_mean.append(float(match.group(1)))
        #    b_max.append(float(match.group(2)))
        ##d_line = log_lines[i + 6]
        ##match = re.search(mean_var, d_line)
        ##if match:
        ##    d_mean.append(float(match.group(1)))
        ##    d_max.append(float(match.group(2)))
       
        if "fc_fusion" in log_lines[i]:        
            print(i)
            #match = pattern.search(line)
            #if 'cases' not in line:    
            #    # 将字符串转换为字典
            #    entry = json.loads(line)
            #else:
            #    if match:
            #        json_part = match.group(1)
            #        entry = json.loads(json_part)
            entry = json.loads(line)
            # 检查是否为"fc_fusion"操作
            if entry.get("op") == "fc_fusion":
                # 提取参数
                params = entry.get("params", {})
                desc = entry.get("desc", {})
                
                # 分别提取a, b, c, d的rows和cols，并存储在各自的列表中
                if 'a' in params:
                    if params['a']['trans']:
                        list_a.append([params['a']["cols"], params['a']["rows"]])
                    else:
                        list_a.append([params['a']["rows"], params['a']["cols"]])
                if 'b' in params:
                    if params['b']['trans']:    
                        list_b.append([params['b']["cols"], params['b']["rows"]])
                    else:
                        list_b.append([params['b']["rows"], params['b']["cols"]])
                if 'c' in params:
                    if params['c']['trans']:
                        list_c.append([params['c']["cols"], params['c']["rows"]])
                    else:
                        list_c.append([params['c']["rows"], params['c']["cols"]])
                if 'd' in params:
                    if params['d']['trans']:
                        list_d.append([params['d']["cols"], params['d']["rows"]])
                    else:
                        list_d.append([params['d']["rows"], params['d']["cols"]])
                
                # 提取alpha和beta，转换为float，并存储
                alpha = float(desc.get("alpha", 0))
                beta = float(desc.get("beta", 0))
                alpha_beta.append([alpha, beta])
except FileNotFoundError:
    print(f"Error: The file {log_file_path} does not exist.")
except IOError as e:
    print(f"Error: An I/O error occurred while reading {log_file_path}: {str(e)}")


unique_string = []
u_list_a = []
u_list_b = []
u_a_m = []
u_a_v = []
u_b_m = []
u_b_v = []

for i in range(len(a_mean)):
    a_m = get_rounded_num(a_mean[i])
    b_m = get_rounded_num(b_mean[i])
    a_v = get_rounded_num(a_max[i])
    b_v = get_rounded_num(b_max[i])
    new_string = str(list_a[i]) + str(list_b[i]) + str(a_m) + str(b_m) + str(a_v) + str(b_v)
    if new_string not in unique_string:
        unique_string.append(new_string)
        u_list_a.append(list_a[i])
        u_list_b.append(list_b[i])
        u_a_m.append(a_m)
        u_b_m.append(b_m)
        u_a_v.append(a_v)
        u_b_v.append(b_v)

print("-------------------------------------------", len(u_list_a))

#list_a = u_list_a
#list_b = u_list_b
#a_mean = u_a_m
#b_mean = u_b_m
#a_max = u_a_v
#b_max = u_b_v


### 输出各个列表
print("List A:", list_a)
print("List B:", list_b)
print("List C:", list_c)
print("List D:", list_d)
print("Alpha and Beta:", alpha_beta)
#print("a_mean", a_mean)
#print("b_mean", b_mean)
#print("d_mean", d_mean)
#print("a_max", a_max)
#print("b_max", b_max)
#print("d_max", d_max)
#
#print("-------------------------------------------", len(a_mean))
#
if DEVICE == 'XPU' and CREATE_DATA:
    for i in range(len(list_a)):
        print(i)
    #for i in range(1):
        x = create_random_tensor(list_a[i], dtype, int(generate_true_random_number(4)))
        w = create_random_tensor(list_b[i], dtype, int(generate_true_random_number(4)))
        bias = create_random_tensor([list_b[i][1]], dtype, int(generate_true_random_number(4)))
        out_grad = create_random_tensor([list_a[i][0], list_b[i][1]], dtype, int(generate_true_random_number(4)))
        paddle.save([x, w, bias, out_grad], input_file + '/linear_' + str(i))


for i in range(len(list_a)):
#for i in range(1):
    print(i)
    x, w, bias, out_grad = paddle.load('FC_INPUT/linear_' + str(i))
    w = w.cast(calculate_type)
    bias = bias.cast(calculate_type)
    x = x.cast(calculate_type)
    linear = Linear(w.shape[0], w.shape[1], bias_attr=True)
    linear.weight.set_value(w)
    linear.bias.set_value(bias)
    linear.train()
    
    out = linear(x)
    paddle.save([True, [out]], out_file + '/linear_' + str(i))
    out = paddle.cast(out, "float32")
    paddle.autograd.backward(tensors=[out], grad_tensors=[out_grad])
    paddle.save([True, [linear.weight.grad, linear.bias.grad]], out_back_file + '/linear_' + str(i))
