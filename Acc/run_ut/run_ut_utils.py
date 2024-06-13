hf_32_standard_api = ["conv1d", "conv2d"]


class Backward_Message:
    MULTIPLE_BACKWARD_MESSAGE = "Multiple backward is not supported."
    UNSUPPORT_BACKWARD_MESSAGE = "function with out=... arguments don't support automatic differentiation, skip backward."
    NO_BACKWARD_RESULT_MESSAGE = "function backward result is None, skip backward."

import os
import yaml
from file_check_util import FileOpen
cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(os.path.dirname(cur_path),"configs","white_list.yaml")
WhiteList = []
with FileOpen(yaml_path, 'r') as f:
    Ops = yaml.safe_load(f)
    WhiteList = Ops.get('white_list')
