import os
import numpy as np
import paddle
import stat
import json
import re
import random
from .file_check_util import check_link, check_file_suffix, FileOpen
from .logger import print_error_log
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker


class Const:
    """
    Class for const
    """

    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FILE_PATTERN = r"^[a-zA-Z0-9_./-]+$"
    MODEL_TYPE = [".onnx", ".pb", ".om"]
    SEMICOLON = ";"
    COLON = ":"
    EQUAL = "="
    COMMA = ","
    DOT = "."
    DUMP_RATIO_MAX = 100
    SUMMERY_DATA_NUMS = 256
    ONE_HUNDRED_MB = 100 * 1024 * 1024
    FLOAT_EPSILON = np.finfo(float).eps
    SUPPORT_DUMP_MODE = ["api", "acl"]
    ON = "ON"
    OFF = "OFF"
    BACKWARD = "backward"
    FORWARD = "forward"
    DELIMITER = "."
    FLOAT_TYPE = [
        np.half,
        np.single,
        float,
        np.double,
        np.float64,
        np.longdouble,
        np.float32,
        np.float16,
    ]
    BOOL_TYPE = [bool, np.uint8]
    INT_TYPE = [np.int32, np.int64]

    # dump mode
    ALL = "all"
    LIST = "list"
    RANGE = "range"
    STACK = "stack"
    ACL = "acl"
    API_LIST = "api_list"
    API_STACK = "api_stack"
    DUMP_MODE = [ALL, LIST, RANGE, STACK, ACL, API_LIST, API_STACK]

    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

    RAISE_PRECISION_PADDLE = {
        paddle.float16: paddle.float32,
        paddle.bfloat16: paddle.float32,
        paddle.float32: paddle.float64,
    }
    CONVERT = {
        "int32_to_int64": ["paddle.int32", "paddle.int64"],
    }

    CONVERT_API = {"int32_to_int64": ["cross_entropy"]}


class FileCheckConst:
    """
    Class for file check const
    """

    READ_ABLE = "read"
    WRITE_ABLE = "write"
    READ_WRITE_ABLE = "read and write"
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FILE_VALID_PATTEN = r"^[a-zA-Z0-9_.:/-]+$"
    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    JSON_SUFFIX = ".json"
    PT_SUFFIX = ".pt"
    CSV_SUFFIX = ".csv"
    YAML_SUFFIX = ".yaml"
    MAX_PKL_SIZE = 1 * 1024 * 1024 * 1024
    MAX_NUMPY_SIZE = 10 * 1024 * 1024 * 1024
    MAX_JSON_SIZE = 1 * 1024 * 1024 * 1024
    MAT_PT_SIZE = 10 * 1024 * 1024 * 1024
    MAX_CSV_SIZE = 1 * 1024 * 1024 * 1024
    MAX_YAML_SIZE = 10 * 1024 * 1024 * 1024
    DIR = "dir"
    FILE = "file"
    DATA_DIR_AUTHORITY = 0o750
    DATA_FILE_AUTHORITY = 0o640
    FILE_SIZE_DICT = {
        PKL_SUFFIX: MAX_PKL_SIZE,
        NUMPY_SUFFIX: MAX_NUMPY_SIZE,
        JSON_SUFFIX: MAX_JSON_SIZE,
        PT_SUFFIX: MAT_PT_SIZE,
        CSV_SUFFIX: MAX_CSV_SIZE,
        YAML_SUFFIX: MAX_YAML_SIZE,
    }


class CompareException(Exception):
    """
    Class for Accuracy Compare Exception
    """

    NONE_ERROR = 0
    INVALID_PATH_ERROR = 1
    OPEN_FILE_ERROR = 2
    CLOSE_FILE_ERROR = 3
    READ_FILE_ERROR = 4
    WRITE_FILE_ERROR = 5
    INVALID_FILE_ERROR = 6
    PERMISSION_ERROR = 7
    INDEX_OUT_OF_BOUNDS_ERROR = 8
    NO_DUMP_FILE_ERROR = 9
    INVALID_DATA_ERROR = 10
    INVALID_PARAM_ERROR = 11
    INVALID_DUMP_RATIO = 12
    INVALID_DUMP_FILE = 13
    UNKNOWN_ERROR = 14
    INVALID_DUMP_MODE = 15
    PARSE_FILE_ERROR = 16
    INVALID_COMPARE_MODE = 17

    def __init__(self, code, error_info: str = ""):
        super(CompareException, self).__init__()
        self.code = code
        self.error_info = error_info

    def __str__(self):
        return self.error_info


def check_object_type(check_object, allow_type):
    if not isinstance(check_object, allow_type):
        print_error_log(f"{check_object} not of {allow_type} type")
        raise CompareException(CompareException.INVALID_DATA_ERROR)


def check_file_or_directory_path(path, isdir=False):
    if isdir:
        if not os.path.exists(path):
            print_error_log("The path {} is not exist.".format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.path.isdir(path):
            print_error_log("The path {} is not a directory.".format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.access(path, os.W_OK):
            print_error_log(
                "The path {} does not have permission to write. Please check the path permission".format(
                    path
                )
            )
            raise CompareException(CompareException.INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            print_error_log("{} is an invalid file or non-exist.".format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

    if not os.access(path, os.R_OK):
        print_error_log(
            "The path {} does not have permission to read. Please check the path permission".format(
                path
            )
        )
        raise CompareException(CompareException.INVALID_PATH_ERROR)


def check_file_size(input_file, max_size):
    try:
        file_size = os.path.getsize(input_file)
    except OSError as os_error:
        print_error_log('Failed to open "%s". %s' % (input_file, str(os_error)))
        raise CompareException(CompareException.INVALID_FILE_ERROR) from os_error
    if file_size > max_size:
        print_error_log(
            "The size (%d) of %s exceeds (%d) bytes, tools not support."
            % (file_size, input_file, max_size)
        )
        raise CompareException(CompareException.INVALID_FILE_ERROR)


def create_directory(dir_path):
    try:
        os.makedirs(dir_path, mode=FileCheckConst.DATA_DIR_AUTHORITY, exist_ok=True)
    except OSError as ex:
        print_error_log(
            "Failed to create {}. Please check the path permission or disk space. {}".format(
                dir_path, str(ex)
            )
        )
        raise CompareException(CompareException.INVALID_PATH_ERROR) from ex


def get_json_contents(file_path):
    ops = get_file_content_bytes(file_path)
    try:
        json_obj = json.loads(ops)
    except ValueError as error:
        print_error_log('Failed to load "%s". %s' % (file_path, str(error)))
        raise CompareException(CompareException.INVALID_FILE_ERROR) from error
    if not isinstance(json_obj, dict):
        print_error_log("Json file %s, content is not a dictionary!" % file_path)
        raise CompareException(CompareException.INVALID_FILE_ERROR)
    return json_obj


def get_file_content_bytes(file):
    with FileOpen(file, "rb") as file_handle:
        return file_handle.read()


def check_need_convert(api_name):
    convert_type = None
    for key, value in Const.CONVERT_API.items():
        if api_name not in value:
            continue
        else:
            convert_type = key
    return convert_type


def get_full_data_path(data_path, real_data_path):
    if not data_path:
        return data_path
    full_data_path = os.path.join(real_data_path, data_path)
    return os.path.realpath(full_data_path)


def check_path_before_create(path):
    if (
        len(os.path.realpath(path)) > Const.DIRECTORY_LENGTH
        or len(os.path.basename(path)) > Const.FILE_NAME_LENGTH
    ):
        print_error_log("The file path length exceeds limit.")
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    if not re.match(Const.FILE_PATTERN, os.path.realpath(path)):
        print_error_log("The file path {} contains special characters.".format(path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)


def api_json_read(input_file):
    check_link(input_file)
    forward_file = os.path.realpath(input_file)
    check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
    forward_content = {}
    if input_file:
        check_link(input_file)
        forward_file = os.path.realpath(input_file)
        check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
        forward_content = get_json_contents(forward_file)
    return forward_content


def seed_all(seed=1234, dist=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    # 分布式场景需额外加上
    if dist:
        global_seed, local_seed = seed, seed
        tracker = get_rng_state_tracker()
        try:
            tracker.add("global_seed", global_seed)
            tracker.add("local_seed", local_seed)
        except Exception as err:
            print('paddle tracker.add("global_seed",global_seed)', str(err))


def parse_args(args):
    tensor_list = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            data_list = parse_args(arg)
            for item in data_list:
                tensor_list.append(item)
        elif isinstance(arg, dict):
            data_list = parse_dict(arg)
            for item in data_list:
                tensor_list.append(item)
        elif isinstance(arg, paddle.Tensor):
            if not arg.stop_gradient:
                tensor_list.append(arg)
        else:
            continue
    return tensor_list


def parse_dict(dict_data):
    ret_tensor_list = []
    for _, value in dict_data.items():
        if value is None:
            continue
        elif isinstance(value, paddle.Tensor):
            if not value.stop_gradient:
                ret_tensor_list.append(value)
    return ret_tensor_list


def check_grad_list(grad_list):
    all_none = True
    valid_varibale_list = []
    for grad in grad_list:
        if not isinstance(grad, type(None)):
            valid_varibale_list.append(grad)
            all_none = False
    if all_none:
        return None
    else:
        return valid_varibale_list
