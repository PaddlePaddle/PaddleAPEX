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

try:
    import paddlenlp
except:
    print("paddlenlp not imported")

from .. import config
from .get_target_op import GetTargetOP
from .OPTemplate import OPTemplate, HookOp

cfg = config.cfg


def wrapped_op(op_name):
    def op_template(*args, **kwargs):
        return OPTemplate(op_name)(*args, **kwargs)
    return op_template


def try_import(package_str="paddle"):
    try:
        MODULE = __import__(package_str)
        input(f"Import {package_str} success")
        return MODULE
    except ImportError as err:
        print(f"Import {package_str} failed, error message is {err}")
        return None

def hijack_api():
    op = GetTargetOP(cfg.op_target_pth)
    target_op = op.get_target_ops()
    # package = []
    # for item in target_op:
    #     package.append(item.split('.')[0])
    # package = set(package)
    # for pack in package:
    #     try_import(pack)
    for op_name in target_op:
        parent_package, method_name = op_name.rsplit('.', maxsplit=1)
        try:
            # pack = package.append(parent_package.split('.')[0])
            # MODULE = try_import(pack)
            setattr(HookOp, "wrap_" + op_name, getattr(eval(parent_package), method_name))
        except Exception as err:
            print(op_name, str(err))

    for attr_name in dir(HookOp):
        if attr_name.startswith("wrap_"):
            parent_package, method_name = attr_name[5:].rsplit('.', maxsplit=1)
            # print(f"parent_package: {parent_package}; method_name: {method_name}")
            setattr(eval(parent_package), method_name, wrapped_op(attr_name[5:]))


            