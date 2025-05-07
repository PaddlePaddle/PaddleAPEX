import paddle
import paddle.nn as nn
import json
import numpy as np
import os
import re

try:
    from paddle_xpu.layers.nn import Linear
except ImportError:
    from paddle.nn import Linear

import json
CREATE_DATA = False
#DEVICE = 'XPU'
DEVICE = 'GPU'

input_file = 'FC_INPUT'
xpu_file = 'XPU'
cal_out_file = xpu_file + '/output'
cal_out_back_file = xpu_file + '/output_backward'
base_out_file = 'BASE/output'
base_out_back_file = 'BASE/output_backward'

base_type = paddle.float32
dtype = paddle.bfloat16
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

for i in range(len(os.listdir(input_file))):
#for i in range(1):
    print(i)
    x, w, bias, out_grad = paddle.load(input_file + '/linear_' + str(i))
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
