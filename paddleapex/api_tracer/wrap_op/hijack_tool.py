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


import inspect
from .. import config
from ...utils import try_import
from .get_target_op import GetTargetOP, GetTargetCls
from .OPTemplate import OPTemplate, HookOp, temp_init, temp_forward

cfg = config.cfg


def wrapped_op(op_name):
    def op_template(*args, **kwargs):
        return OPTemplate(op_name)(*args, **kwargs)

    return op_template

# def wrapped_cls(cls_name, cls):
#     def op_template(*args, **kwargs):
#         # 可以从HookCls中获取cls对象
#         wrap_class = type(f"{cls_name}", (cls,), {'__init__': temp_init, 'forward': temp_forward})
#         return wrap_class(cls_name, *args, **kwargs)

#     return op_template

def add_class(cls, parent_package, class_name, whole_name):
    # print(whole_name)
    module = eval(parent_package)

    original_class = getattr(module, class_name, None)
    if original_class:
        print(f"Original class before replacement: {class_name} (id: {id(original_class)})")
    else:
        print(f"No original class named {class_name} found in {parent_package}")
    
    wrap_class = type(f"{whole_name}", (cls,), {'__init__': temp_init, 'forward': temp_forward}) 
    setattr(module, class_name, wrap_class)

    new_class = getattr(module, class_name, None)
    if new_class:
        print(f"New class after replacement: {class_name} (id: {id(new_class)})")
    
    if original_class != new_class:
        print(f"Class {class_name} successfully replaced.")
    else:
        print(f"Class {class_name} replacement failed.")



def hijack_api():
    op = GetTargetOP(cfg.op_target_pth)
    cls = GetTargetCls(cfg.cls_target_pth)
    target_op = op.get_target_ops()
    target_cls = cls.get_target_ops()
    for op_name in target_op:
        parent_package, method_name = op_name.rsplit(".", maxsplit=1)
        try:
            pack = parent_package.split(".")[0]
            package_name, module = try_import(pack)
            globals()[package_name] = module
            target_object = getattr(eval(parent_package), method_name)
            if inspect.isclass(target_object):
                raise ValueError(f"{op_name} is a class")
            else:
                setattr(
                    HookOp, "wrap_" + op_name, target_object
                )          
        except Exception as err:
            print(op_name, str(err))

    for cls_name in target_cls:
        parent_package, method_name = cls_name.rsplit(".", maxsplit=1)
        try:
            pack = parent_package.split(".")[0]
            package_name, module = try_import(pack)
            globals()[package_name] = module
            target_object = getattr(eval(parent_package), method_name)
            if inspect.isclass(target_object):
                add_class(target_object, parent_package, method_name, cls_name)
            else:
                raise ValueError(f"{cls} not a class")
        except Exception as err:
            print(cls_name, str(err))
    
    for attr_name in dir(HookOp):
        if attr_name.startswith("wrap_"):
            parent_package, method_name = attr_name[5:].rsplit(".", maxsplit=1)
            setattr(eval(parent_package), method_name, wrapped_op(attr_name[5:]))
