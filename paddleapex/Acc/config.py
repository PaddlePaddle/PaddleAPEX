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
        # Load environment variable, if user did not set, tool load from predefined default setting.
        config_path = os.environ.get('APEX_CONFIG_PATH','/Users/xujinming/Desktop/Reconstruct_project/PaddleAPEX/paddleapex/Acc/configs/tool_config.yaml')#
        with open(config_path, "r", encoding="utf-8") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            self.dump_mode = configs["dump_mode"]
            self.op_target_pth = configs["op_target_path"]
            self.dump_root_path = configs["dump_root_path"]
            self.target_step = configs["target_step"]
            self.remote_path = configs['remote_path']
            self.Async_dump = configs['Async_dump']
            f.close()

        self.global_step = -1
        self.dump_state = False
        self.Op_count = {}
        self.prefix_op_name_ = None

    def new_step(self):
        self.global_step += 1
        if self.global_step in self.target_step:
            self.Op_count = {}
            self.dump_state = True
        else:
            self.dump_state = False

cfg = Config()
