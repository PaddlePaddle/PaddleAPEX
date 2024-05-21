import paddle.tensor
import yaml
import paddle
from .. import config

cfg = config.cfg

class GetTargetOP:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            Ops = yaml.safe_load(f)
            self.WrapTensorOps = Ops.get("tensor")
            self.WrapFunctionalOps = Ops.get("functional")
            self.WrapPaddleOps = Ops.get("paddle")
            f.close()
        self.WhiteTensorOps = None
        self.WhiteFunctionalOps = None
        self.WhitePaddleOps = None

        if cfg.white_list:
            with open(cfg.white_list, "r") as fw:
                Ops = yaml.safe_load(fw)
                self.WhiteTensorOps = Ops.get("tensor")
                self.WhiteFunctionalOps = Ops.get("functional")
                self.WhitePaddleOps = Ops.get("paddle")
            fw.close()

    def get_target_ops(self, api_type):
        if api_type == "Tensor":
            _all_tensor_ops = dir(paddle.Tensor)
            target_op = set(_all_tensor_ops) & set(self.WrapTensorOps)
            if self.WhiteTensorOps:
                print(f"Tensor api:{self.WhiteTensorOps} are ignored in Acc tool!")
                target_op = set(target_op) - set(self.WhiteTensorOps)
        elif api_type == "functional":
            _all_functional_ops = dir(paddle.nn.functional)
            target_op = set(_all_functional_ops) & set(self.WrapFunctionalOps)
            if self.WhiteFunctionalOps:
                print(f"Functional api:{self.WhiteFunctionalOps} are ignored in Acc tool!")
                target_op = set(target_op) - set(self.WhiteFunctionalOps)
        elif api_type == "paddle":
            _all_paddle_ops = dir(paddle)
            target_op = set(_all_paddle_ops) & set(self.WrapPaddleOps)
            if self.WhitePaddleOps:
                print(f"Paddle api:{self.WhitePaddleOps} are ignored in Acc tool!")
                target_op = set(target_op) - set(self.WhitePaddleOps)
        else:
            print(api_type, " is not a vlid api type!")
        return target_op
