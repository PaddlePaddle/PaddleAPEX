import paddle
import paddle.nn as nn

import os

def generate_true_random_number(num_bytes):
    """ 生成真随机数，从 /dev/urandom 读取指定数量的字节 """
    # 从 /dev/urandom 读取 num_bytes 个字节
    random_bytes = os.urandom(num_bytes)
    # 将字节转换为整数
    return int.from_bytes(random_bytes, 'big')

# 生成一个真随机数
#random_number = generate_true_random_number(4)  # 读取 4 bytes
#print(random_number)

from paddleapex import Tracer
checker = Tracer()
checker.register_op()

checker.start()

paddle.seed(42)
q_len = [131328, 147728, 106128, 128128, 120128]
#q_len = [10880, 12224, 1280, 1424, 15104, 1520, 1664, 16976, 18224, 1856, 1952, 20096, 22592, 23840, 24320, 2432, 27344, 29360, 30080, 8192, 9200, 9872]
#for i in q_len:
for i in range(20):
    paddle.seed(int(generate_true_random_number(4)))
    q = paddle.rand((1, 147728, 30, 64), dtype=paddle.bfloat16)
    paddle.seed(int(generate_true_random_number(6)))
    k = paddle.rand((1, 147728, 30, 64), dtype=paddle.bfloat16)
    paddle.seed(int(generate_true_random_number(5)))
    v = paddle.rand((1, 147728, 30, 64), dtype=paddle.bfloat16)
    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False

    output = paddle.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False, True)
    output.stop_gradient = False
    output.backward()


#q = paddle.rand((80, 1408, 32, 64), dtype=paddle.bfloat16)
#k = paddle.rand((80, 1408, 32, 64), dtype=paddle.bfloat16)
#v = paddle.rand((80, 1408, 32, 64), dtype=paddle.bfloat16)
#q.stop_gradient = False
#k.stop_gradient = False
#v.stop_gradient = False
#
#output = paddle.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False, True)
#
#dout = paddle.zeros_like(output)
#output.backward(dout)

#q_len = [131328, 147728, 106128, 128128, 120128]
#for i in q_len:
#	q = paddle.rand((1, i, 30, 64), dtype=paddle.bfloat16)
#	k = paddle.rand((1, i, 30, 64), dtype=paddle.bfloat16)
#	v = paddle.rand((1, i, 30, 64), dtype=paddle.bfloat16)
#	output = paddle.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False, True)
#
checker.stop()
print(output)
