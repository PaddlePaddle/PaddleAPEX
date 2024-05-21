import paddle
from .utils.get_target_op import GetTargetOP
from . import config
from .wrap_Tensor_op import TensorOPTemplate, HookTensorOp
from .wrap_functional_op import FunctionalOPTemplate, HookFunctionalOp
from .wrap_paddle_op import PaddleOPTemplate, HookPaddleOp


cfg = config.cfg

def wrapped_op(api_type, op_name):
    if api_type == "Tensor":

        def tensor_op_template(*args, **kwargs):
            return TensorOPTemplate(op_name)(*args, **kwargs)

        return tensor_op_template
    elif api_type == "functional":

        def functional_op_template(*args, **kwargs):
            return FunctionalOPTemplate(op_name)(*args, **kwargs)

        return functional_op_template
    elif api_type == "paddle":

        def paddle_op_template(*args, **kwargs):
            return PaddleOPTemplate(op_name)(*args, **kwargs)

        return paddle_op_template
    else:
        print("In func wrapped_op:", api_type, " is not a vlid api type!")
        return None


def hijack_tensor_api(target_op):
    for op_name in target_op:
        setattr(HookTensorOp, "wrap_" + op_name, getattr(paddle.Tensor, str(op_name)))
    for attr_name in dir(HookTensorOp):
        if attr_name.startswith("wrap_"):
            setattr(paddle.Tensor, attr_name[5:], wrapped_op("Tensor", attr_name[5:]))


def hijack_functional_api(target_op):
    for op_name in target_op:
        setattr(
            HookFunctionalOp,
            "wrap_" + op_name,
            getattr(paddle.nn.functional, str(op_name)),
        )
    for attr_name in dir(HookFunctionalOp):
        if attr_name.startswith("wrap_"):
            setattr(
                paddle.nn.functional,
                attr_name[5:],
                wrapped_op("functional", attr_name[5:]),
            )


def hijack_paddle_api(target_op):
    for op_name in target_op:
        setattr(HookPaddleOp, "wrap_" + op_name, getattr(paddle, str(op_name)))

    for attr_name in dir(HookPaddleOp):
        if attr_name.startswith("wrap_"):
            setattr(paddle, attr_name[5:], wrapped_op("paddle", attr_name[5:]))


def hijack_target_api():
    op = GetTargetOP(cfg.op_target_pth)
    hijack_tensor_api(op.get_target_ops("Tensor"))
    hijack_functional_api(op.get_target_ops("functional"))
    hijack_paddle_api(op.get_target_ops("paddle"))
