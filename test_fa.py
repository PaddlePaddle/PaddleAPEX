import paddle
import paddle.nn as nn
import json

import os

def generate_true_random_number(num_bytes):
    """ 生成真随机数，从 /dev/urandom 读取指定数量的字节 """
    # 从 /dev/urandom 读取 num_bytes 个字节
    random_bytes = os.urandom(num_bytes)
    # 将字节转换为整数
    #print(int.from_bytes(random_bytes, 'big'))
    return int.from_bytes(random_bytes, 'big')


# 路径到你的日志文件
log_file_path = 'fa.log'

# 初始化存储列表
lod_seqlens_q_list = []
head_num_list = []
head_dim_list = []
max_seqlen_q_list = []

def convert_to_list(s):
    # 去除字符串中的花括号
    s = s.strip('{}')
    # 分割字符串
    elements = s.split(',')
    # 将字符串元素转换为整数，并返回列表
    return [int(element.strip()) for element in elements]

# 读取和解析日志文件
try:
    with open(log_file_path, 'r') as file:
        for line in file:
            try:
                # 将每行转换为JSON对象
                data = json.loads(line)
                # 检查是否是目标操作
                if data.get('op') == 'mha_varlen_bwd':
                    params = data.get('params', {})
                    seqlens_q = params.get('lod_seqlens_q', [])
                    #lod_seqlens_q_list.append(params.get('lod_seqlens_q', []))
                    lod_seqlens_q_list.append(convert_to_list(seqlens_q))
                    head_num_list.append(params.get('head_num', None))
                    head_dim_list.append(params.get('head_dim', None))
                    max_seqlen_q_list.append(params.get('max_seqlen_q', None))
            except json.JSONDecodeError:
                print("Warning: Failed to decode JSON from line:", line)
except FileNotFoundError:
    print(f"Error: The file {log_file_path} does not exist.")
except IOError as e:
    print(f"Error: An I/O error occurred while reading {log_file_path}: {str(e)}")



from paddleapex import Tracer
checker = Tracer()
checker.register_op()

paddle.seed(46)
B = []
for b in lod_seqlens_q_list:
    if len(b) == 2:
        B.append(1)
    if len(b) > 2:
        B.append(int(b[-1] / b[1]))
L = max_seqlen_q_list
H = head_num_list
D = head_dim_list

# 输出结果
print("LOD Sequence Lengths Q:", B)
print("Head Numbers:", H)
print("Head Dimensions:", D)
print("Max Sequence Length Q:", L)


for i in range(len(B)):
    paddle.seed(int(generate_true_random_number(4)))
    q = paddle.rand((B[i], L[i], H[i], D[i]), dtype=paddle.bfloat16)
    paddle.seed(int(generate_true_random_number(4)))
    k = paddle.rand((B[i], L[i], H[i], D[i]), dtype=paddle.bfloat16)
    paddle.seed(int(generate_true_random_number(4)))
    v = paddle.rand((B[i], L[i], H[i], D[i]), dtype=paddle.bfloat16)
    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False

    output = paddle.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False, True)
    output.stop_gradient = False
    output.backward()

#for i in range(10):
#    paddle.seed(int(generate_true_random_number(4)))
#    q = paddle.rand((1, 147728, 30, 64), dtype=paddle.bfloat16)
#    paddle.seed(int(generate_true_random_number(6)))
#    k = paddle.rand((1, 147728, 30, 64), dtype=paddle.bfloat16)
#    paddle.seed(int(generate_true_random_number(5)))
#    v = paddle.rand((1, 147728, 30, 64), dtype=paddle.bfloat16)
#    q.stop_gradient = False
#    k.stop_gradient = False
#    v.stop_gradient = False
#
#    output = paddle.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False, True)
#    output.stop_gradient = False
#    output.backward()

checker.stop()


