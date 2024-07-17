import paddle
from paddleapex import Tracer

if __name__ == "__main__":
    a = paddle.randn([2,2])
    b = paddle.randn([2,2])
    a.stop_gradient = False
    b.stop_gradient = False
    apex = Tracer()
    apex.start()
    y = paddle.add(a,b)
    y = paddle.multiply(y,b)
    z = y ** 2
    z.backward()
    apex.stop()