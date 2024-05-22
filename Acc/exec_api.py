import paddle
import paddle.nn.functional as F


def exec(op_name, api_type):
    if "unction" in api_type:
        return getattr(F, op_name)
    elif "addle" in api_type:
        return getattr(paddle, op_name)
    elif "Tensor" in api_type:
        return getattr(paddle.Tensor, op_name)
    else:
        print("In Exec: Undefined api type!")


if __name__ == "__main__":
    tensor_x = paddle.randn([8192, 5120])
    tensor_x.to("bfloat16")

    args = [tensor_x]
    kwargs = {"axes": [0], "starts": [1024], "ends": [2048]}
    # 传入方法名称和分类，有paddle、functional、tensor 大小写不区分.
    ret = exec("slice", "paddle")(*args, **kwargs)
    print(ret.shape)
