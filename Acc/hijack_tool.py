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
from .utils.get_target_op import GetTargetOP
from . import config
from .wrap_Tensor_op import TensorOPTemplate, HookTensorOp
from .wrap_functional_op import FunctionalOPTemplate, HookFunctionalOp
from .wrap_paddle_op import PaddleOPTemplate, HookPaddleOp
from .wrap_custom_op import CustomOPTemplate, HookCustomOp
import sys
import os
import importlib
import importlib.util
cfg = config.cfg


def check_module(module_name):
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None:
        print("Module :{} not found".format(module_name))
    else:
        print("Module:{} can be imported!".format(module_name))

def wrapped_op(api_type, op_name):
    if api_type == "Tensor":

        def tensor_op_template(*args, **kwargs):
            return TensorOPTemplate(op_name)(*args, **kwargs)

        return tensor_op_template
    elif api_type == "functional":

        def functional_op_template(*args, **kwargs):
            return FunctionalOPTemplate(op_name)(*args, **kwargs)

        return functional_op_template
    elif api_type == "paddle":

        def paddle_op_template(*args, **kwargs):
            return PaddleOPTemplate(op_name)(*args, **kwargs)

        return paddle_op_template
    elif api_type == "custom":

        def custom_op_template(*args, **kwargs):
            return CustomOPTemplate(op_name)(*args, **kwargs)

        return custom_op_template
    else:
        print("In func wrapped_op:", api_type, " is not a vlid api type!")
        return None

def hijack_custom_api(target_op):
    hijack_file = cfg.cutom_op_file_path
    custom_abs_pth = os.path.abspath(hijack_file)
    custom_abs_dir = os.path.dirname(custom_abs_pth)
    sys.path.append(custom_abs_dir)
    module_name = custom_abs_pth.split("/")[-1][:-3]
    check_module(module_name)
    CUSTOM_MODULE = importlib.import_module(module_name)
    print(dir(CUSTOM_MODULE))
    for op_name in target_op:
        setattr(HookCustomOp, "wrap_" + op_name, getattr(CUSTOM_MODULE, str(op_name)))
    for attr_name in dir(HookCustomOp):
        if attr_name.startswith("wrap_"):
            setattr(CUSTOM_MODULE, attr_name[5:], wrapped_op("Custom", attr_name[5:]))

def hijack_tensor_api(target_op):
    for op_name in target_op:
        setattr(HookTensorOp, "wrap_" + op_name, getattr(paddle.Tensor, str(op_name)))
    for attr_name in dir(HookTensorOp):
        if attr_name.startswith("wrap_"):
            setattr(paddle.Tensor, attr_name[5:], wrapped_op("Tensor", attr_name[5:]))


def hijack_functional_api(target_op):
    for op_name in target_op:
        setattr(
            HookFunctionalOp,
            "wrap_" + op_name,
            getattr(paddle.nn.functional, str(op_name)),
        )
    for attr_name in dir(HookFunctionalOp):
        if attr_name.startswith("wrap_"):
            setattr(
                paddle.nn.functional,
                attr_name[5:],
                wrapped_op("functional", attr_name[5:]),
            )


def hijack_paddle_api(target_op):
    for op_name in target_op:
        setattr(HookPaddleOp, "wrap_" + op_name, getattr(paddle, str(op_name)))

    for attr_name in dir(HookPaddleOp):
        if attr_name.startswith("wrap_"):
            setattr(paddle, attr_name[5:], wrapped_op("paddle", attr_name[5:]))


def hijack_target_api():
    op = GetTargetOP(cfg.op_target_pth)
    hijack_tensor_api(op.get_target_ops("Tensor"))
    hijack_functional_api(op.get_target_ops("functional"))
    hijack_paddle_api(op.get_target_ops("paddle"))

    if cfg.custom_op:
        custom_op = GetTargetOP(cfg.custom_op_path)
    hijack_custom_api(custom_op.get_target_ops("custom"))
