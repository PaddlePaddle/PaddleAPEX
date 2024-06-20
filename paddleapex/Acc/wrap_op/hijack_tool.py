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

import paddle

from .. import config
from .get_target_op import GetTargetOP
from .OPTemplate import OPTemplate, HookOp

cfg = config.cfg


def wrapped_op(op_name):
    def op_template(*args, **kwargs):
        return OPTemplate(op_name)(*args, **kwargs)
    return op_template

def hijack_api():
    op = GetTargetOP(cfg.op_target_pth)
    target_op = op.get_target_ops()
    for op_name in target_op:
        parent_package, method_name = op_name.rsplit('.', maxsplit=1)
        try:
            setattr(HookOp, "wrap_" + op_name, getattr(eval(parent_package), method_name))
        except Exception as err:
            print(op_name, str(err))

    for attr_name in dir(HookOp):
        if attr_name.startswith("wrap_"):
            setattr(eval(parent_package), method_name, wrapped_op(attr_name[5:]))