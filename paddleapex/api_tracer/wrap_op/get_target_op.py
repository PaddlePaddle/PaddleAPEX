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

import yaml
from paddleapex.api_tracer import config
from paddleapex.utils import try_import

cfg = config.cfg


class GetTargetOP:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            Ops = yaml.safe_load(f)
            self.target_op = Ops.get("target_op")
            self.ignored_op = Ops.get("ignored_op")
            self.target_class = Ops.get("target_class")
            f.close()
            if self.ignored_op is None:
                self.ignored_op = []
            self.api_to_catch = set(self.target_op) - set(self.ignored_op)

    def check_api_stack(self):
        for api in self.api_to_catch:
            try:
                pack = api.split(".")[0]
                package_name, module = try_import(pack)
                globals()[package_name] = module
                func = eval(api)
                if not func:
                    print(f"{api} is not available!")
            except Exception as err:
                print(f"For api: {api}   ", str(err))

    def get_target_ops(self):
        self.api_to_catch = set(self.target_op) - set(self.ignored_op)
        if cfg.profile_mode:
            self.api_to_catch -= set(["paddle.max", "paddle.min"])
        self.check_api_stack()
        return self.api_to_catch

    def get_target_class(self):
        return self.target_class
