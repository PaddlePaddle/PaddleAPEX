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
from paddleapex.api_tracer.Dump import dump_util
from paddleapex.api_tracer.wrap_op.hijack_tool import hijack_api
from paddleapex.api_tracer.config import cfg


class Tracer:
    def __init__(self):
        hijack_api()

    def start(self):
        # global step counting.
        cfg.new_step()

    def stop(self):
        if cfg.dump_state:
            dump_util.dump()
