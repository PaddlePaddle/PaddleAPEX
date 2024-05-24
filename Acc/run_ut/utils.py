import os
import numpy as np
import paddle
import stat
import time
import sys


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
