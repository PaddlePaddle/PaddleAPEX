import os
import yaml

class Config:
    def __init__(self) -> None:
        # Load environment variable, if user not set, tool load from predefined default setting.
        config_path = os.environ.get('CONFIG_PATH','./Acc/configs/tool_config.yaml')#
        with open(config_path, "r", encoding="utf-8") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            self.global_rank = configs["global_rank"]
            self.dump_mode = configs["dump_mode"]
            self.op_target_pth = configs["op_target_path"]
            self.dump_root_path = configs["dump_root_path"]
            self.target_step = configs["target_step"]
            if configs["white_list"]!= "None":
                self.white_list = configs["white_list"]
            else:
                self.white_list = None
            f.close()

        self.global_step = -1
        self.dump_state = False
        self.Paddle_op_count = {}
        self.Tensor_op_count = {}
        self.Functional_op_count = {}

        self.prefix_paddle_op_name_ = None
        self.prefix_functional_op_name_ = None
        self.prefix_tensor_op_name_ = None

    def new_step(self):
        self.global_step += 1
        if self.global_step in self.target_step:
            self.Paddle_op_count = {}
            self.Tensor_op_count = {}
            self.Functional_op_count = {}
            self.Paddletensor_op_count = {}
            self.prefix_paddle_op_name_ = None
            self.prefix_functional_op_name_ = None
            self.prefix_tensor_op_name_ = None
            self.prefix_Paddletensor_op_name_ = None
            self.dump_state = True
        else:
            self.dump_state = False

cfg = Config()
