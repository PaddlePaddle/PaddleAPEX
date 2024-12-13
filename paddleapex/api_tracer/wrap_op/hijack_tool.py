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

from .. import config
from ...utils import try_import
from .get_target_op import GetTargetOP
from .OPTemplate import OPTemplate, HookOp, hijack_call

cfg = config.cfg

def wrapped_op(op_name):
    def op_template(*args, **kwargs):
        return OPTemplate(op_name)(*args, **kwargs)

    return op_template


def hijack_api():
    op = GetTargetOP(cfg.op_target_pth)
    target_op = op.get_target_ops()
    target_class = op.get_target_class()
    for op_name in target_op:
        parent_package, method_name = op_name.rsplit(".", maxsplit=1)
        try:
            pack = parent_package.split(".")[0]
            package_name, module = try_import(pack)
            globals()[package_name] = module
            setattr(
                HookOp, "wrap_" + op_name, getattr(eval(parent_package), method_name)
            )
        except Exception as err:
            print(op_name, str(err))
    
    for class_in in target_class:
        # print("begin class --------------------------------", class_in)
        parent_package, class_n = class_in.rsplit(".", maxsplit=1)
        try:
            class_name, model = try_import(parent_package)
            model = getattr(model, class_n)
            model.prefix_op_name_ = class_in
            model.__call__ = hijack_call
        except Exception as err:
            print(class_in, str(err))

    for attr_name in dir(HookOp):
        if attr_name.startswith("wrap_"):
            parent_package, method_name = attr_name[5:].rsplit(".", maxsplit=1)
            setattr(eval(parent_package), method_name, wrapped_op(attr_name[5:]))
