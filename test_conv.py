import paddle
import paddle.nn as nn
import json
import numpy as np
import os
import re


dtype = paddle.bfloat16
paddle.set_default_dtype(dtype)

log_file_path = 'conv.log'
pattern = re.compile(r".*?cases(.*)")

# 初始化列表
list_n = []
list_c = []
list_xh = []
list_xw = []
list_f = []
list_ksize = []
list_stride = []
list_pad = []
list_dilation = []
list_group = []
list_nchw = []

n = 0
try:
    with open(log_file_path, 'r') as file:
        for line in file:
            n = n + 1
            print(n)
            entry = json.loads(line)
            # 检查是否为"fc_fusion"操作
            if entry.get("op") == "cudnn_conv2d_grad":
                # 提取参数
                params = entry.get("params", {})
                types = entry.get("desc", {})
                
                if 'n' in params:
                    list_n.append(params['n'])
                if 'c' in params:
                    list_c.append(params['c'])
                if 'xh' in params:
                    list_xh.append(params['xh'])
                elif 'h' in params:
                    list_xh.append(params['h'])
                if 'xw' in params:
                    list_xw.append(params['xw'])
                elif 'w' in params:
                    list_xw.append(params['w'])
                if 'f' in params:
                    list_f.append(params['f'])
                if '_ksize' in params:
                    list_ksize.append(params['_ksize'])
                if '_stride' in params:
                    list_stride.append(params['_stride'])
                if '_pad' in params:
                    list_pad.append(params['_pad'])
                if '_dilation' in params:
                    list_dilation.append(params['_dilation'])
                if 'group' in params:
                    list_group.append(params['group'])
                if 'is_nchw' in params:
                    list_nchw.append(params['is_nchw'])
except FileNotFoundError:
    print(f"Error: The file {log_file_path} does not exist.")
except IOError as e:
    print(f"Error: An I/O error occurred while reading {log_file_path}: {str(e)}")

def generate_true_random_number(num_bytes):
    """ 生成真随机数，从 /dev/urandom 读取指定数量的字节 """
    # 从 /dev/urandom 读取 num_bytes 个字节
    random_bytes = os.urandom(num_bytes)
    # 将字节转换为整数
    #print(int.from_bytes(random_bytes, 'big'))
    return int.from_bytes(random_bytes, 'big')

CREATE_DATA = False
DEVICE = 'XPU'
#DEVICE = 'GPU'

# 输出各个列表
print("List n:", list_n)
print("List c:", list_c)
print("List xh:", list_xh)
print("List xw:", list_xw)
print("List f:", list_f)
print("List ksize:", list_ksize)
print("List stride:", list_stride)
print("List pad:", list_pad)
print("List dilation:", list_dilation)
print("List group:", list_group)
print("List nchw:", list_nchw)

x_shape = []
f_shape = []
y_shape = []
for i in range(len(list_n)):
    n = list_n[i]
    c = list_c[i]
    xh = list_xh[i]
    xw = list_xw[i]
    f = list_f[i]
    ksize = list_ksize[i]
    pad = list_pad[i]
    dilation = list_dilation[i]
    stride = list_stride[i]

    x_shape.append([n, c, xh, xw])
    f_shape.append([ksize[0], ksize[1], c, f])
    h_out = (xh + 2 * pad[0] - (dilation[0] * (ksize[0] - 1) + 1)) / stride[0] + 1
    w_out = (xw + 2 * pad[1] - (dilation[1] * (ksize[1] - 1) + 1)) / stride[1] + 1
    y_shape.append([n, f, int(h_out), int(w_out)])

print(x_shape)
print(f_shape)
print(y_shape)

dtype = paddle.bfloat16
paddle.set_default_dtype(dtype)
#
if DEVICE == 'XPU' and CREATE_DATA:
    for i in range(len(list_n)):
    #for i in range(1):
        paddle.seed(int(generate_true_random_number(4)))
        x = np.random.uniform(-1, 1, x_shape[i]).astype("float32")
        paddle.seed(int(generate_true_random_number(5)))
        f = np.random.uniform(-1, 1, f_shape[i]).astype("float32")
        paddle.seed(int(generate_true_random_number(6)))
        out_grad = np.random.uniform(-1, 1, y_shape[i]).astype("float32")
        x = paddle.to_tensor(x, stop_gradient=False).cast(dtype)
        f = paddle.to_tensor(f, stop_gradient=False).cast(dtype)
        out_grad = paddle.to_tensor(out_grad, stop_gradient=True).cast(dtype)
        paddle.save([x, f, out_grad], 'CONV_INPUT/conv_' + str(i))

for i in range(len(list_n)):
#for i in range(1):
    x, f, out_grad = paddle.load('CONV_INPUT/conv_' + str(i))
    conv = nn.Conv2D(list_c[i], list_f[i], kernel_size=list_ksize[i], 
            stride=list_stride[i], padding=list_pad[i], dilation=list_dilation[i], 
            groups=list_group[i])
    if list_group[i] == 1:
        conv.weight.set_value(f)
    conv.train()
    
    out = conv(x)
    #paddle.save([True, [out]], 'XPU/output/conv_' + str(i))
    #out = paddle.cast(out, "float32")
    paddle.autograd.backward(tensors=[out], grad_tensors=[out_grad])
    #paddle.save([True, [linear.weight.grad, linear.bias.grad]], 'XPU/output_backward/linear_' + str(i))
#
#
#x = paddle.uniform((1, 1, 1, 51200), dtype=dtype, min=-1., max=1.)
#x.stop_gradient = False
#conv = nn.Conv2D(1, 512, kernel_size=[1, 10], stride=[1, 5], padding=[0,0,0,0], dilation=[1,1], groups=1)
#y = conv(x)
#print(y.shape)
#y.stop_gradient = False
#out_grad = paddle.uniform(y.shape, dtype=dtype, min=-1., max=1.)
#paddle.autograd.backward(tensors=[y], grad_tensors=[out_grad])
#print(x.grad)
