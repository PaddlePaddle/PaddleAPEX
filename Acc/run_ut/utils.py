import os
import numpy as np
import paddle
import stat
import time
import sys
import json
import re
import random
from file_check_util import FileOpen
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker


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
    DELIMITER = '.'
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

    RAISE_PRECISION_PADDLE = {
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
        YAML_SUFFIX: MAX_YAML_SIZE
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


def _print_log(level, msg, end='\n'):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getgid()
    print(current_time + "(" + str(pid) + ")-[" + level + "]" + msg, end=end)
    sys.stdout.flush()


def print_error_log(error_msg):
    """
    Function Description:
        print error log.
    Parameter:
        error_msg: the error message.
    """
    _print_log("ERROR", error_msg)


def print_info_log(info_msg, end='\n'):
    """
    Function Description:
        print info log.
    Parameter:
        info_msg: the info message.
    """
    _print_log("INFO", info_msg, end=end)


def print_warn_log(warn_msg):
    """
    Function Description:
        print warn log.
    Parameter:
        warn_msg: the warning message.
    """
    _print_log("WARNING", warn_msg)


def check_object_type(check_object, allow_type):
    """
    Function Description:
        Check if the object belongs to a certain data type
    Parameter:
        check_object: the object to be checked
        allow_type: legal data type
    Exception Description:
        when invalid data throw exception
    """
    if not isinstance(check_object, allow_type):
        print_error_log(f"{check_object} not of {allow_type} type")
        raise CompareException(CompareException.INVALID_DATA_ERROR)


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


def _check_pkl(pkl_file_handle, file_name):
    tensor_line = pkl_file_handle.readline()
    if len(tensor_line) == 0:
        print_error_log("dump file {} have empty line!".format(file_name))
        raise CompareException(CompareException.INVALID_DUMP_FILE)
    pkl_file_handle.seek(0, 0)


def check_file_mode(npu_pkl, bench_pkl, stack_mode):
    npu_pkl_name = os.path.split(npu_pkl)[-1]
    bench_pkl_name = os.path.split(bench_pkl)[-1]

    if not npu_pkl_name.startswith("api_stack") and not bench_pkl_name.startswith("api_stack"):
        if stack_mode:
            print_error_log("The current file does not contain stack information, please turn off the stack_mode")
            raise CompareException(CompareException.INVALID_COMPARE_MODE)
    elif npu_pkl_name.startswith("api_stack") and bench_pkl_name.startswith("api_stack"):
        if not stack_mode:
            print_error_log("The current file contains stack information, please turn on the stack_mode")
            raise CompareException(CompareException.INVALID_COMPARE_MODE)
    else:
        print_error_log("The dump mode of the two files is not same, please check the dump files")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)


def check_file_size(input_file, max_size):
    try:
        file_size = os.path.getsize(input_file)
    except OSError as os_error:
        print_error_log('Failed to open "%s". %s' % (input_file, str(os_error)))
        raise CompareException(CompareException.INVALID_FILE_ERROR) from os_error
    if file_size > max_size:
        print_error_log('The size (%d) of %s exceeds (%d) bytes, tools not support.'
                        % (file_size, input_file, max_size))
        raise CompareException(CompareException.INVALID_FILE_ERROR)


def get_dump_data_path(dump_dir):
    """
    Function Description:
        traverse directories and obtain the absolute path of dump data
    Parameter:
        dump_dir: dump data directory
    Return Value:
        dump data path,file is exist or file is not exist
    """
    dump_data_path = None
    file_is_exist = False

    check_file_or_directory_path(dump_dir, True)
    for dir_path, sub_paths, files in os.walk(dump_dir):
        if len(files) != 0:
            dump_data_path = dir_path
            file_is_exist = True
            break
        dump_data_path = dir_path
    return dump_data_path, file_is_exist


def modify_dump_path(dump_path, mode):
    if mode == Const.ALL:
        return dump_path
    file_name = os.path.split(dump_path)
    mode_file_name = mode + "_" + file_name[-1]
    return os.path.join(file_name[0], mode_file_name)


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


def save_numpy_data(file_path, data):
    """
    save_numpy_data
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    np.save(file_path, data)


def parse_arg_value(values):
    """
    parse dynamic arg value of atc cmdline
    """
    value_list = []
    for item in values.split(Const.SEMICOLON):
        value_list.append(parse_value_by_comma(item))
    return value_list


def parse_value_by_comma(value):
    """
    parse value by comma, like '1,2,4,8'
    """
    value_list = []
    value_str_list = value.split(Const.COMMA)
    for value_str in value_str_list:
        value_str = value_str.strip()
        if value_str.isdigit() or value_str == '-1':
            value_list.append(int(value_str))
        else:
            print_error_log("please check your input shape.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return value_list


def get_data_len_by_shape(shape):
    data_len = 1
    for item in shape:
        if item == -1:
            print_error_log("please check your input shape, one dim in shape is -1.")
            return -1
        data_len = data_len * item
    return data_len


def add_time_as_suffix(name):
    return '{}_{}.csv'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))


def format_value(value):
    return '{:.6f}'.format(value)


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

def get_file_content_bytes(file):
    with FileOpen(file, 'rb') as file_handle:
        return file_handle.read()


def check_need_convert(api_name):
    convert_type = None
    for key, value in Const.CONVERT_API.items():
        if api_name not in value:
            continue
        else:
            convert_type = key
    return convert_type


def api_info_preprocess(api_name, api_info_dict):
    """
    Function Description:
        Preprocesses the API information.
    Parameter:
        api_name: Name of the API.
        api_info_dict: argument of the API.
    Return api_info_dict:
        convert_type: Type of conversion.
        api_info_dict: Processed argument of the API.
    """
    convert_type = check_need_convert(api_name)
    if api_name == 'cross_entropy':
        api_info_dict = cross_entropy_process(api_info_dict)
    return convert_type, api_info_dict


def cross_entropy_process(api_info_dict):
    """
    Function Description:
        Preprocesses the cross_entropy API information.
    Parameter:
        api_info_dict: argument of the API.
    Return api_info_dict:
        api_info_dict: Processed argument of the API.
    """
    if 'args' in api_info_dict and len(api_info_dict['args']) > 1 and 'Min' in api_info_dict['args'][1]:
        if api_info_dict['args'][1]['Min'] <= 0:
            # The second argument in cross_entropy should be -100 or not less than 0
            api_info_dict['args'][1]['Min'] = 0
    return api_info_dict


def get_full_data_path(data_path, real_data_path):
    if not data_path:
        return data_path
    full_data_path = os.path.join(real_data_path, data_path)
    return os.path.realpath(full_data_path)


def check_path_before_create(path):
    if len(os.path.realpath(path)) > Const.DIRECTORY_LENGTH or len(os.path.basename(path)) > \
            Const.FILE_NAME_LENGTH:
        print_error_log('The file path length exceeds limit.')
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    if not re.match(Const.FILE_PATTERN, os.path.realpath(path)):
        print_error_log('The file path {} contains special characters.'.format(path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    # 分布式场景需额外加上
    global_seed, local_seed = seed,seed # 这样ok?
    tracker = get_rng_state_tracker()
    try:
        tracker.add("global_seed",global_seed)
        tracker.add("local_seed",local_seed)
    except Exception as err:
        print('paddle tracker.add("global_seed",global_seed)', str(err))
