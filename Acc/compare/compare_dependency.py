# config.yaml和hook_module.support_wrap_ops.yaml这两个配置文件需要统一

import numpy as np
import os
import stat
import time
import sys
import yaml
import re
import json
import random
import csv

import paddle

def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    # 分布式场景需额外加上
    global_seed, local_seed = seed,seed # 这样ok?
    tracker = paddle.get_rng_state_tracker()
    # tracker = paddle.distributed.fleet.meta_parallel.get_rng_state_tracker()
    try:
        tracker.add("global_seed",global_seed)
        tracker.add("local_seed",local_seed)
    except Exception as err:
        print('paddle tracker.add("global_seed",global_seed)', str(err))

class Const:
    """
    Class for const
    """
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    MODEL_TYPE = ['.onnx', '.pb', '.om']
    SEMICOLON = ";"
    COLON = ":"
    EQUAL = "="
    COMMA = ","
    DOT = "."
    DUMP_RATIO_MAX = 100
    SUMMERY_DATA_NUMS = 256
    ONE_HUNDRED_MB = 100 * 1024 * 1024
    FLOAT_EPSILON = np.finfo(float).eps
    SUPPORT_DUMP_MODE = ['api', 'acl']
    ON = 'ON'
    OFF = 'OFF'
    BACKWARD = 'backward'
    FORWARD = 'forward'
    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble, np.float32, np.float16]
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

    RAISE_PRECISION = {
        paddle.float16: paddle.float32,
        paddle.bfloat16: paddle.float32,
        paddle.float32: paddle.float64
    }
    CONVERT = {
        "int32_to_int64": ["paddle.int32", "paddle.int64"],
    }

    CONVERT_API = {
        "int32_to_int64": ["cross_entropy"]
    }

def write_csv(data, filepath):
    with FileOpen(filepath, 'a', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def get_file_content_bytes(file):
    with FileOpen(file, 'rb') as file_handle:
        return file_handle.read()

def get_json_contents(file_path):
    ops = get_file_content_bytes(file_path)
    try:
        json_obj = json.loads(ops)
    except ValueError as error:
        print_error_log('Failed to load "%s". %s' % (file_path, str(error)))
        raise CompareException(CompareException.INVALID_FILE_ERROR) from error
    if not isinstance(json_obj, dict):
        print_error_log('Json file %s, content is not a dictionary!' % file_path)
        raise CompareException(CompareException.INVALID_FILE_ERROR)
    return json_obj

def check_file_or_directory_path(path, isdir=False):
    """
    Function Description:
        check whether the path is valid
    Parameter:
        path: the path to check
        isdir: the path is dir or file
    Exception Description:
        when invalid data throw exception
    """
    if isdir:
        if not os.path.exists(path):
            print_error_log('The path {} is not exist.'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.path.isdir(path):
            print_error_log('The path {} is not a directory.'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.access(path, os.W_OK):
            print_error_log(
                'The path {} does not have permission to write. Please check the path permission'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            print_error_log('{} is an invalid file or non-exist.'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

    if not os.access(path, os.R_OK):
        print_error_log(
            'The path {} does not have permission to read. Please check the path permission'.format(path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)

def create_directory(dir_path):
    """
    Function Description:
        creating a directory with specified permissions in a thread-safe manner
    Parameter:
        dir_path: directory path
    Exception Description:
        when invalid data throw exception
    """
    try:
        os.makedirs(dir_path, mode=FileCheckConst.DATA_DIR_AUTHORITY, exist_ok=True)
    except OSError as ex:
        print_error_log(
            'Failed to create {}. Please check the path permission or disk space. {}'.format(dir_path, str(ex)))
        raise CompareException(CompareException.INVALID_PATH_ERROR) from ex

def check_path_before_create(path):
    if len(os.path.realpath(path)) > Const.DIRECTORY_LENGTH or len(os.path.basename(path)) > \
            Const.FILE_NAME_LENGTH:
        print_error_log('The file path length exceeds limit.')
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    if not re.match(Const.FILE_PATTERN, os.path.realpath(path)):
        print_error_log('The file path {} contains special characters.'.format(path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)

def change_mode(path, mode):
    if not os.path.exists(path) or os.path.islink(path):
        return
    try:
        os.chmod(path, mode)
    except PermissionError as ex:
        print_error_log('Failed to change {} authority. {}'.format(path, str(ex)))
        raise FileCheckException(FileCheckException.INVALID_PERMISSION_ERROR) from ex

def _print_log(level, msg, end='\n'):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getgid()
    print(current_time + "(" + str(pid) + ")-[" + level + "]" + msg, end=end)
    sys.stdout.flush()


def print_info_log(info_msg, end='\n'):
    """
    Function Description:
        print info log.
    Parameter:
        info_msg: the info message.
    """
    _print_log("INFO", info_msg, end=end)


def print_error_log(error_msg):
    """
    Function Description:
        print error log.
    Parameter:
        error_msg: the error message.
    """
    _print_log("ERROR", error_msg)


def print_warn_log(warn_msg):
    """
    Function Description:
        print warn log.
    Parameter:
        warn_msg: the warning message.
    """
    _print_log("WARNING", warn_msg)

class FileCheckConst:
    """
    Class for file check const
    """
    READ_ABLE = "read"
    WRITE_ABLE = "write"
    READ_WRITE_ABLE = "read and write"
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FILE_VALID_PATTERN = r"^[a-zA-Z0-9_.:/-]+$"
    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    JSON_SUFFIX = ".json"
    PT_SUFFIX = ".pt"
    CSV_SUFFIX = ".csv"
    YAML_SUFFIX = ".yaml"
    MAX_PKL_SIZE = 1 * 1024 * 1024 * 1024
    MAX_NUMPY_SIZE = 10 * 1024 * 1024 * 1024
    MAX_JSON_SIZE = 1 * 1024 * 1024 * 1024
    MAX_PT_SIZE = 10 * 1024 * 1024 * 1024
    MAX_CSV_SIZE = 1 * 1024 * 1024 * 1024
    MAX_YAML_SIZE = 10 * 1024 * 1024
    DIR = "dir"
    FILE = "file"
    DATA_DIR_AUTHORITY = 0o750
    DATA_FILE_AUTHORITY = 0o640
    FILE_SIZE_DICT = {
        PKL_SUFFIX: MAX_PKL_SIZE,
        NUMPY_SUFFIX: MAX_NUMPY_SIZE,
        JSON_SUFFIX: MAX_JSON_SIZE,
        PT_SUFFIX: MAX_PT_SIZE,
        CSV_SUFFIX: MAX_CSV_SIZE,
        YAML_SUFFIX: MAX_YAML_SIZE
    }


class FileCheckException(Exception):
    """
    Class for File Check Exception
    """
    NONE_ERROR = 0
    INVALID_PATH_ERROR = 1
    INVALID_FILE_TYPE_ERROR = 2
    INVALID_PARAM_ERROR = 3
    INVALID_PERMISSION_ERROR = 3

    def __init__(self, code, error_info: str = ""):
        super(FileCheckException, self).__init__()
        self.code = code
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class FileChecker:
    """
    The class for check file.

    Attributes:
        file_path: The file or dictionary path to be verified.
        path_type: file or dictionary
        ability(str): FileCheckConst.WRITE_ABLE or FileCheckConst.READ_ABLE to set file has writability or readability
        file_type(str): The correct file type for file
    """
    def __init__(self, file_path, path_type, ability=None, file_type=None, is_script=True):
        self.file_path = file_path
        self.path_type = self._check_path_type(path_type)
        self.ability = ability
        self.file_type = file_type
        self.is_script = is_script

    @staticmethod
    def _check_path_type(path_type):
        if path_type not in [FileCheckConst.DIR, FileCheckConst.FILE]:
            print_error_log(f'The path_type must be {FileCheckConst.DIR} or {FileCheckConst.FILE}.')
            raise FileCheckException(FileCheckException.INVALID_PARAM_ERROR)
        return path_type

    def common_check(self):
        """
        功能：用户校验基本文件权限：软连接、文件长度、是否存在、读写权限、文件属组、文件特殊字符
        注意：文件后缀的合法性，非通用操作，可使用其他独立接口实现
        """
        check_path_exists(self.file_path)
        check_link(self.file_path)
        self.file_path = os.path.realpath(self.file_path)
        check_path_length(self.file_path)
        check_path_type(self.file_path, self.path_type)
        self.check_path_ability()
        if self.is_script:
            check_path_owner_consistent(self.file_path)
        check_path_pattern_vaild(self.file_path)
        check_common_file_size(self.file_path)
        check_file_suffix(self.file_path, self.file_type)
        return self.file_path

    def check_path_ability(self):
        if self.ability == FileCheckConst.WRITE_ABLE:
            check_path_writability(self.file_path)
        if self.ability == FileCheckConst.READ_ABLE:
            check_path_readability(self.file_path)
        if self.ability == FileCheckConst.READ_WRITE_ABLE:
            check_path_readability(self.file_path)
            check_path_writability(self.file_path)



class FileOpen:
    """
    The class for open file by a safe way.

    Attributes:
        file_path: The file or dictionary path to be opened.
        mode(str): The file open mode
    """
    SUPPORT_READ_MODE = ["r", "rb"]
    SUPPORT_WRITE_MODE = ["w", "wb", "a", "ab"]
    SUPPORT_READ_WRITE_MODE = ["r+", "rb+", "w+", "wb+", "a+", "ab+"]

    def __init__(self, file_path, mode, encoding='utf-8'):
        self.file_path = file_path
        self.mode = mode
        self.encoding = encoding
        self._handle = None

    def __enter__(self):
        self.check_file_path()
        binary_mode = "b"
        if binary_mode not in self.mode:
            self._handle = open(self.file_path, self.mode, encoding=self.encoding)
        else:
            self._handle = open(self.file_path, self.mode)
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle:
            self._handle.close()

    def check_file_path(self):
        support_mode = self.SUPPORT_READ_MODE + self.SUPPORT_WRITE_MODE + self.SUPPORT_READ_WRITE_MODE
        if self.mode not in support_mode:
            print_error_log("File open not support %s mode" % self.mode)
            raise FileCheckException(FileCheckException.INVALID_PARAM_ERROR)
        check_link(self.file_path)
        self.file_path = os.path.realpath(self.file_path)
        check_path_length(self.file_path)
        self.check_ability_and_owner()
        check_path_pattern_vaild(self.file_path)
        if os.path.exists(self.file_path):
            check_common_file_size(self.file_path)

    def check_ability_and_owner(self):
        if self.mode in self.SUPPORT_READ_MODE:
            check_path_exists(self.file_path)
            check_path_readability(self.file_path)
            check_path_owner_consistent(self.file_path)
        if self.mode in self.SUPPORT_WRITE_MODE and os.path.exists(self.file_path):
            check_path_writability(self.file_path)
            check_path_owner_consistent(self.file_path)
        if self.mode in self.SUPPORT_READ_WRITE_MODE and os.path.exists(self.file_path):
            check_path_readability(self.file_path)
            check_path_writability(self.file_path)
            check_path_owner_consistent(self.file_path)


def check_link(path):
    abs_path = os.path.abspath(path)
    if os.path.islink(abs_path):
        print_error_log('The file path {} is a soft link.'.format(path))
        raise FileCheckException(FileCheckException.INVALID_PATH_ERROR)


def check_path_length(path, name_length=None):
    file_max_name_length = name_length if name_length else FileCheckConst.FILE_NAME_LENGTH
    if len(path) > FileCheckConst.DIRECTORY_LENGTH or \
            len(os.path.basename(path)) > file_max_name_length:
        print_error_log('The file path length exceeds limit.')
        raise FileCheckException(FileCheckException.INVALID_PATH_ERROR)


def check_path_exists(path):
    if not os.path.exists(path):
        print_error_log('The file path %s does not exist.' % path)
        raise FileCheckException(FileCheckException.INVALID_PATH_ERROR)


def check_path_readability(path):
    if not os.access(path, os.R_OK):
        print_error_log('The file path %s is not readable.' % path)
        raise FileCheckException(FileCheckException.INVALID_PERMISSION_ERROR)


def check_path_writability(path):
    if not os.access(path, os.W_OK):
        print_error_log('The file path %s is not writable.' % path)
        raise FileCheckException(FileCheckException.INVALID_PERMISSION_ERROR)


def check_path_executable(path):
    if not os.access(path, os.X_OK):
        print_error_log('The file path %s is not executable.' % path)
        raise FileCheckException(FileCheckException.INVALID_PERMISSION_ERROR)


def check_other_user_writable(path):
    st = os.stat(path)
    if st.st_mode & 0o002:
        _user_interactive_confirm(
            'The file path %s may be insecure because other users have write permissions. '
            'Do you want to continue?' % path)


def _user_interactive_confirm(message):
    while True:
        check_message = input(message + " Enter 'c' to continue or enter 'e' to exit: ")
        if check_message == "c":
            break
        elif check_message == "e":
            print_warn_log("User canceled.")
            raise FileCheckException(FileCheckException.INVALID_PATH_ERROR)
        else:
            print("Input is error, please enter 'c' or 'e'.")


def check_path_owner_consistent(path):
    file_owner = os.stat(path).st_uid
    if file_owner != os.getuid():
        print_error_log('The file path %s may be insecure because is does not belong to you.' % path)
        raise FileCheckException(FileCheckException.INVALID_PERMISSION_ERROR)


def check_path_pattern_vaild(path):
    if not re.match(FileCheckConst.FILE_VALID_PATTERN, path):
        print_error_log('The file path %s contains special characters.' % path)
        raise FileCheckException(FileCheckException.INVALID_PATH_ERROR)


def check_file_size(file_path, max_size):
    file_size = os.path.getsize(file_path)
    if file_size >= max_size:
        _user_interactive_confirm(f'The size of file path {file_path} exceeds {max_size} bytes.'
                                  f'Do you want to continue?')


def check_common_file_size(file_path):
    if os.path.isfile(file_path):
        for suffix, max_size in FileCheckConst.FILE_SIZE_DICT.items():
            if file_path.endswith(suffix):
                check_file_size(file_path, max_size)
                break


def check_file_suffix(file_path, file_suffix):
    if file_suffix:
        if not file_path.endswith(file_suffix):
            print_error_log(f"The {file_path} should be a {file_suffix} file!")
            raise FileCheckException(FileCheckException.INVALID_FILE_TYPE_ERROR)


def check_path_type(file_path, file_type):
    if file_type == FileCheckConst.FILE:
        if not os.path.isfile(file_path):
            print_error_log(f"The {file_path} should be a file!")
            raise FileCheckException(FileCheckException.INVALID_FILE_TYPE_ERROR)
    if file_type == FileCheckConst.DIR:
        if not os.path.isdir(file_path):
            print_error_log(f"The {file_path} should be a dictionary!")
            raise FileCheckException(FileCheckException.INVALID_FILE_TYPE_ERROR)

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


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path,"configs","op_target.yaml") # paddle提供的文件
with FileOpen(yaml_path, 'r') as f:
    Ops = yaml.safe_load(f)
    WrapFunctionalOps = Ops.get('functional')
    WrapTensorOps = Ops.get('tensor')
    WrapTorchOps = Ops.get('paddle')

WrapApi = set(WrapFunctionalOps) | set(WrapTensorOps) | set(WrapTorchOps)


class Config:
    def __init__(self, yaml_file):
        check_file_or_directory_path(yaml_file, False)
        with FileOpen(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        self.config = {key: self.validate(key, value) for key, value in config.items()}

    def validate(self, key, value):
        validators = {
            'dump_path': str,
            'real_data': bool,
            'enable_dataloader': bool,
            'target_iter': list,
            'white_list': list,
            'error_data_path': str,
            'jit_compile': bool,
            'precision': int
        }
        if key not in validators:
            raise ValueError(f"{key} must be one of {validators.keys()}")
        if not isinstance(value, validators.get(key)):
            raise ValueError(f"{key} must be {validators[key].__name__} type")
        if key == 'target_iter':
            if not isinstance(value, list):
                raise ValueError("target_iter must be a list type")
            if any(isinstance(i, bool) for i in value):
                raise ValueError("target_iter cannot contain boolean values")
            if not all(isinstance(i, int) for i in value):
                raise ValueError("All elements in target_iter must be of int type")
            if any(i < 0 for i in value):
                raise ValueError("All elements in target_iter must be greater than or equal to 0")
        if key == 'precision' and value < 0:
            raise ValueError("precision must be greater than 0")
        if key == 'white_list':
            if not isinstance(value, list):
                raise ValueError("white_list must be a list type")
            if not all(isinstance(i, str) for i in value):
                raise ValueError("All elements in white_list must be of str type")
            invalid_api = [i for i in value if i not in WrapApi]
            if invalid_api:
                raise ValueError(f"{', '.join(invalid_api)} is not in support_wrap_ops.yaml, please check the white_list")
        return value

    def __getattr__(self, item):
        return self.config[item]

    def __str__(self):
        return '\n'.join(f"{key}={value}" for key, value in self.config.items())

    def update_config(self, dump_path=None, real_data=None, target_iter=None, white_list=None, enable_dataloader=None):
        args = {
            "dump_path": dump_path if dump_path else self.config.get("dump_path", './'),
            "real_data": real_data if real_data else self.config.get("real_data", False),
            "target_iter": target_iter if target_iter else self.config.get("target_iter", [1]),
            "white_list": white_list if white_list else self.config.get("white_list", []),
            "enable_dataloader": enable_dataloader if enable_dataloader else self.config.get("enable_dataloader", False)
        }
        for key, value in args.items():
            if key in self.config:
                self.config[key] = self.validate(key, value)
            else:
                raise ValueError(f"Invalid key '{key}'")


# cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# yaml_path = os.path.join(cur_path, "config.yaml")
yaml_path = os.path.join("config.yaml")
msCheckerConfig = Config(yaml_path)

seed_all()