import time
import os
import numpy as np

import paddle
import yaml

from compare_dependency import Const, print_warn_log, CompareException
from compare_dependency import FileOpen


current_time = time.strftime("%Y%m%d%H%M%S")
API_PRECISION_COMPARE_RESULT_FILE_NAME = "api_precision_compare_result_" + current_time + ".csv"
API_PRECISION_COMPARE_DETAILS_FILE_NAME = "api_precision_compare_details_" + current_time + ".csv"
BENCHMARK_COMPARE_SUPPORT_LIST = ['paddle.float16', 'paddle.bfloat16', 'paddle.float32']
API_PRECISION_COMPARE_UNSUPPORT_LIST = ['paddle.float64', 'paddle.complex64', 'paddle.complex128']
BINARY_COMPARE_UNSUPPORT_LIST = BENCHMARK_COMPARE_SUPPORT_LIST + API_PRECISION_COMPARE_UNSUPPORT_LIST


cur_path = os.path.dirname(os.path.realpath(__file__))
standard_yaml_path = os.path.join(cur_path, "api_precision_standard.yaml")
with FileOpen(standard_yaml_path, 'r') as f:
    Apis = yaml.safe_load(f)
    AbsoluteStandardApi = Apis.get('AbsoluteThreshStandard')
    BinaryStandardApi = Apis.get('BinaryCompareStandard')


threshold_yaml_path = os.path.join(cur_path, "api_precision_threshold.yaml")
with FileOpen(threshold_yaml_path, 'r') as f:
    apis_threshold = yaml.safe_load(f)


DETAIL_TEST_ROWS = [[
            "API Name", "Bench Dtype", "DEVICE Dtype", "Shape",
            "余弦相似度",
            "最大绝对误差",
            "双百指标",
            "双千指标",
            "双万指标",
            "二进制一致错误率",
            "误差均衡性",
            "均方根误差",
            "小值域错误占比",
            "相对误差最大值",
            "相对误差平均值",
            "inf/nan错误率",
            "相对误差错误率",
            "绝对误差错误率",
            "Status",
            "Message"
        ]]


precision_configs = {
    paddle.float16 : {
        'small_value' : [
            1e-3
        ],
        'small_value_atol' : [
            1e-5
        ]
    },
    paddle.bfloat16: {
        'small_value' : [
            1e-3
        ],
        'small_value_atol' : [
            1e-5
        ]
    },
    paddle.float32:{
        'small_value' : [
            1e-6
        ],
        'small_value_atol' : [
            1e-9
        ]
    }
}


class CompareConst:
    NAN = np.nan
    NA = "N/A"
    PASS = 'pass'
    WARNING = 'warning'
    ERROR = 'error'
    SKIP = 'SKIP'
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    BFLOAT16_MIN = -3.3895313892515355e+38
    BFLOAT16_MAX = 3.3895313892515355e+38
    BFLOAT16_EPS = 2 ** -8
    SPACE = " "
    
    
class ApiPrecisionCompareColumn:
    API_NAME = 'API Name'
    DEVICE_DTYPE = 'DEVICE Dtype'
    SMALL_VALUE_ERROR_RATE = '小值域错误占比'
    RMSE = '均方根误差'
    MAX_REL_ERR = '相对误差最大值'
    MEAN_REL_ERR = '相对误差平均值'
    EB = '误差均衡性'
    SMALL_VALUE_ERROR_RATIO = '小值域错误比值'
    SMALL_VALUE_ERROR_STATUS = '小值域判定结果'
    RMSE_RATIO = '均方根误差比值'
    RMSE_STATUS = '均方根误差判定结果'
    MAX_REL_ERR_RATIO = '相对误差最大值比值'
    MAX_REL_ERR_STATUS = '相对误差最大值判定结果'
    MEAN_REL_ERR_RATIO = '相对误差平均值比值'
    MEAN_REL_ERR_STATUS = '相对误差平均值判定结果'
    EB_RATIO = '误差均衡性比值'
    EB_STATUS = '误差均衡性判定结果'
    ERROR_RATE = '二进制一致错误率'
    ERROR_RATE_STATUS = '二进制一致错误率判定结果'
    INF_NAN_ERROR_RATIO = 'inf/nan错误率'
    INF_NAN_ERROR_RATIO_STATUS = 'inf/nan判定结果'
    REL_ERR_RATIO = '相对误差错误率'
    REL_ERR_RATIO_STATUS = '相对误差判定结果'
    ABS_ERR_RATIO = '绝对误差错误率'
    ABS_ERR_RATIO_STATUS = '绝对误差判定结果'
    FINAL_RESULT = '比对结果'
    ALGORITHM = '比对算法'
    FORWWARD_STATUS = 'Forward Test Success'
    BACKWARD_STATUS = 'Backward Test Success'
    MESSAGE = 'Message'
    
    @staticmethod
    def to_required_columns():
        return [ApiPrecisionCompareColumn.API_NAME, ApiPrecisionCompareColumn.DEVICE_DTYPE, 
                ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE, ApiPrecisionCompareColumn.RMSE, 
                ApiPrecisionCompareColumn.MAX_REL_ERR, ApiPrecisionCompareColumn.MEAN_REL_ERR, ApiPrecisionCompareColumn.EB,
                ApiPrecisionCompareColumn.ERROR_RATE, ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO, 
                ApiPrecisionCompareColumn.REL_ERR_RATIO, ApiPrecisionCompareColumn.ABS_ERR_RATIO]

    @staticmethod
    def get_detail_csv_title():
        return [ApiPrecisionCompareColumn.API_NAME,  
                ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATIO, ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_STATUS, 
                ApiPrecisionCompareColumn.RMSE_RATIO, ApiPrecisionCompareColumn.RMSE_STATUS, 
                ApiPrecisionCompareColumn.MAX_REL_ERR_RATIO, ApiPrecisionCompareColumn.MAX_REL_ERR_STATUS, 
                ApiPrecisionCompareColumn.MEAN_REL_ERR_RATIO, ApiPrecisionCompareColumn.MEAN_REL_ERR_STATUS, 
                ApiPrecisionCompareColumn.EB_RATIO, ApiPrecisionCompareColumn.EB_STATUS, 
                ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO, ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO_STATUS, 
                ApiPrecisionCompareColumn.REL_ERR_RATIO, ApiPrecisionCompareColumn.REL_ERR_RATIO_STATUS, 
                ApiPrecisionCompareColumn.ABS_ERR_RATIO, ApiPrecisionCompareColumn.ABS_ERR_RATIO_STATUS, 
                ApiPrecisionCompareColumn.ERROR_RATE, ApiPrecisionCompareColumn.ERROR_RATE_STATUS, 
                ApiPrecisionCompareColumn.FINAL_RESULT, ApiPrecisionCompareColumn.ALGORITHM, ApiPrecisionCompareColumn.MESSAGE]
    
    @staticmethod
    def get_result_csv_title():
        return [ApiPrecisionCompareColumn.API_NAME, ApiPrecisionCompareColumn.FORWWARD_STATUS, 
                ApiPrecisionCompareColumn.BACKWARD_STATUS, ApiPrecisionCompareColumn.MESSAGE]


CompareMessage = {
    "topk" : "在npu上，topk的入参sorted=False时不生效，会返回有序tensor，而cpu上会返回无序tensor。 如果topk精度不达标，请检查是否是该原因导致的。"
}


def check_dtype_comparable(x, y):
    if x.dtype in Const.FLOAT_TYPE:
        if y.dtype in Const.FLOAT_TYPE:
            return True 
        return False 
    if x.dtype in Const.BOOL_TYPE:
        if y.dtype in Const.BOOL_TYPE:
            return True 
        return False 
    if x.dtype in Const.INT_TYPE:
        if y.dtype in Const.INT_TYPE:
            return True 
        return False
    print_warn_log(f"Compare: Unexpected dtype {x.dtype}, {y.dtype}")
    return False


def convert_str_to_float(input_data):
    if isinstance(input_data, str) and input_data.strip() == "":
        msg = 'ERROR: Input data is an empty string'
        raise CompareException(CompareException.INVALID_DATA_ERROR, msg)
    try:
        float_data = float(input_data)
        if str(float_data) in ('inf', '-inf', 'nan'):
            msg = 'ERROR: Input data is either "inf", "-inf", "nan"'
            raise CompareException(CompareException.INVALID_DATA_ERROR, msg)
        return float_data
    except ValueError as e:
        msg = 'ERROR: Input data cannot be converted to float'
        raise CompareException(CompareException.INVALID_DATA_ERROR, msg) from e
        