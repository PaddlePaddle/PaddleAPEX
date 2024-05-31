# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml

class Config:
    def __init__(self) -> None:
        # Load environment variable, if user not set, tool load from predefined default setting.
        config_path = os.environ.get('APEX_CONFIG_PATH','./PaddleAPEX/Acc/configs/tool_config.yaml')#
        with open(config_path, "r", encoding="utf-8") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            self.dump_mode = configs["dump_mode"]
            self.op_target_pth = configs["op_target_path"]
            self.dump_root_path = configs["dump_root_path"]
            self.target_step = configs["target_step"]
            self.remote_path = configs['remote_path']
            self.Async_dump = configs['Async_dump']
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
