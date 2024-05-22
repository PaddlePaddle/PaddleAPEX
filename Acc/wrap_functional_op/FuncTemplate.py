from ..api_info import API
from .. import config
import paddle.nn as nn
import paddle.distributed as dist


class HookFunctionalOp:
    pass


cfg = config.cfg


class FunctionalOPTemplate(nn.Layer):
    def __init__(self, op_name):
        self.op_name_ = op_name
        cfg.prefix_functional_op_name_ = "Functional*" + str(op_name) + "*"

    def forward(self, *inputs, **kwargs):

        if self.op_name_ not in cfg.Functional_op_count:
            cfg.Functional_op_count[self.op_name_] = 1
            cfg.prefix_functional_op_name_ += "0"
        else:
            cfg.Functional_op_count[self.op_name_] += 1
            cfg.prefix_functional_op_name_ += str(
                cfg.Functional_op_count[self.op_name_] - 1
            )

        if cfg.dump_state:
            api_recorder = API(cfg.dump_mode)
            rank = dist.get_rank()
            api_recorder.update_APIInfo(config.prefix_functional_op_name_, rank)
            output = getattr(HookFunctionalOp, "wrap_" + str(self.op_name_))(
                *inputs, **kwargs
            )
            api_recorder.update_real_data(output, inputs, kwargs)
        else:
            output = getattr(HookFunctionalOp, "wrap_" + str(self.op_name_))(
                *inputs, **kwargs
            )
        return output

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)
