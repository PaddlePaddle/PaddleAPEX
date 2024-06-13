import argparse
import os
import sys
import csv
import math
from collections import namedtuple
import paddle
import pandas as pd

from compare.compare_utils import CompareConst, API_PRECISION_COMPARE_RESULT_FILE_NAME, \
API_PRECISION_COMPARE_DETAILS_FILE_NAME, BENCHMARK_COMPARE_SUPPORT_LIST, API_PRECISION_COMPARE_UNSUPPORT_LIST, \
    ApiPrecisionCompareColumn, AbsoluteStandardApi, BinaryStandardApi, ULPStandardApi, ThousandthStandardApi, \
    BINARY_COMPARE_UNSUPPORT_LIST, ULP_COMPARE_SUPPORT_LIST, convert_str_to_float, CompareMessage, is_inf_or_nan
from compare.compare_column import ApiPrecisionOutputColumn
from compare.compare_dependency import get_validated_result_csv_path

from compare.compare_dependency import print_info_log, print_warn_log, print_error_log, write_csv, CompareException, create_directory
from compare.compare_dependency import FileCheckConst, FileChecker, change_mode, check_path_before_create

PRECISION = 14

CompareConfig = namedtuple('CompareConfig', ['npu_csv_path', 'gpu_csv_path', 'result_csv_path', 'details_csv_path'])
unsupported_message = 'This data type does not support benchmark compare.'


benchmark_algorithms_thresholds = {
    'small_value' : {
        'error_threshold' : 2,
        'warning_threshold' : 1
    },
    'rmse' : {
        'error_threshold' : 2,
        'warning_threshold' : 1
    },
    'max_rel_err' : {
        'error_threshold' : 10,
        'warning_threshold' : 1
    },
    'mean_rel_err' : {
        'error_threshold' : 2,
        'warning_threshold' : 1
    },
    'eb' : {
        'error_threshold' : 2,
        'warning_threshold' : 1
    }
}


benchmark_message = {
    "small_value_err_status": {
        CompareConst.ERROR: "ERROR: 小值域错误比值超过阈值\n",
        CompareConst.WARNING: "WARNING: 小值域错误比值超过阈值\n"
    },
    "rmse_status": {
        CompareConst.ERROR: "ERROR: 均方根误差比值超过阈值\n",
        CompareConst.WARNING: "WARNING: 均方根误差比值超过阈值\n"
    },
    "max_rel_err_status": {
        CompareConst.ERROR: "ERROR: 相对误差最大值比值超过阈值\n",
        CompareConst.WARNING: "WARNING: 相对误差最大值比值超过阈值\n"
    },
    "mean_rel_err_status": {
        CompareConst.ERROR: "ERROR: 相对误差平均值比值超过阈值\n",
        CompareConst.WARNING: "WARNING: 相对误差平均值比值超过阈值\n"
    }
}


class Standard:
    @staticmethod
    def _calc_ratio(column_name, x, y, default_value):
        '''
        计算npu侧和gpu侧统计量的比值
        输入：
            column_name：统计量名称
            x：npu侧统计量
            y：gpu侧统计量
            default：当x不接近0，y接近0，设置的比值默认值
        输出： 
            ratio：统计量x和y的比值
            inf_nan_consistency：不出现inf或nan时为True，出现inf或nan时必须同时为inf或-inf或nan才为True，否则为False
            message：当出现inf或nan时的提示信息
        '''
        x, y = convert_str_to_float(x), convert_str_to_float(y)
        if is_inf_or_nan(x) or is_inf_or_nan(y):
            if math.isnan(x) or math.isnan(y):
                if math.isnan(x) and math.isnan(y):
                    return float("nan"), True, f"{column_name}同为同号inf或nan\n"
                else:
                    return float("nan"), False, f"{column_name}inf或nan不一致\n"
            else:
                if math.isinf(x) and math.isinf(y):
                    if x == y:
                        return float("nan"), True, f"{column_name}同为同号inf或nan\n"
                    else:
                        return float("nan"), False, f"{column_name}inf或nan不一致\n"
                elif math.isinf(x):
                    if y >= 0:
                        return x, False, f"{column_name}inf或nan不一致\n"
                    else:
                        return -x, False, f"{column_name}inf或nan不一致\n"
                else:
                    return abs(x / y), False, f"{column_name}inf或nan不一致\n"
        inf_nan_consistency = True
        message = ""
        if math.isclose(y, 0.0):
            if math.isclose(x, 0.0):
                return 1.0, inf_nan_consistency, message
            else:
                return default_value, inf_nan_consistency, message
        else:
            return abs(x / y), inf_nan_consistency, message


class BenchmarkStandard(Standard):
    def __init__(self, api_name, npu_precision, gpu_precision):
        self.api_name = api_name
        self.npu_precision = npu_precision
        self.gpu_precision = gpu_precision
        self.small_value_err_ratio = 1
        self.rmse_ratio = 1
        self.max_rel_err_ratio = 1
        self.mean_rel_err_ratio = 1
        self.eb_ratio = 1
        self.small_value_err_status = CompareConst.PASS
        self.rmse_status = CompareConst.PASS
        self.max_rel_err_status = CompareConst.PASS
        self.mean_rel_err_status = CompareConst.PASS
        self.eb_status = CompareConst.PASS
        self.check_result_list = []
        self.final_result = CompareConst.PASS
        self.compare_message = ""

    def __str__(self):
        return f"{self.api_name}"

    def get_result(self):
        small_value_inf_nan_consistency, rmse_inf_nan_consistency, \
        max_rel_inf_nan_consistency, mean_rel_inf_nan_consistency, eb_inf_nan_consistency = self._compare_ratio()
        if small_value_inf_nan_consistency:
            self.small_value_err_status = self._get_status(self.small_value_err_ratio, 'small_value')
        else:
            self.small_value_err_status = CompareConst.ERROR
        self.check_result_list.append(self.small_value_err_status)
        if rmse_inf_nan_consistency:
            self.rmse_status = self._get_status(self.rmse_ratio, 'rmse')
        else:
            self.rmse_status = CompareConst.ERROR
        self.check_result_list.append(self.rmse_status)
        if max_rel_inf_nan_consistency:
            self.max_rel_err_status = self._get_status(self.max_rel_err_ratio, 'max_rel_err')
        else:
            self.max_rel_err_status = CompareConst.ERROR
        self.check_result_list.append(self.max_rel_err_status)
        if mean_rel_inf_nan_consistency:
            self.mean_rel_err_status = self._get_status(self.mean_rel_err_ratio, 'mean_rel_err')
        else:
            self.mean_rel_err_status = CompareConst.ERROR
        self.check_result_list.append(self.mean_rel_err_status)
        if eb_inf_nan_consistency:
            self.eb_status = self._get_status(self.eb_ratio, 'eb')
        else:
            self.eb_status = CompareConst.ERROR
        if CompareConst.ERROR in self.check_result_list:
            self.final_result = CompareConst.ERROR
        elif CompareConst.WARNING in self.check_result_list:
            self.final_result = CompareConst.WARNING

    def _compare_ratio(self):
        self.small_value_err_ratio, small_value_inf_nan_consistency, small_value_message = self._calc_ratio(ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE,
            self.npu_precision.get(ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE),
            self.gpu_precision.get(ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE), 10000.0)
        self.compare_message += small_value_message
        self.rmse_ratio, rmse_inf_nan_consistency, rmse_message = self._calc_ratio(ApiPrecisionCompareColumn.RMSE,
                                           self.npu_precision.get(ApiPrecisionCompareColumn.RMSE),
                                            self.gpu_precision.get(ApiPrecisionCompareColumn.RMSE), 10000.0)
        self.compare_message += rmse_message
        self.max_rel_err_ratio, max_rel_inf_nan_consistency, max_rel_message = self._calc_ratio(ApiPrecisionCompareColumn.MAX_REL_ERR,
                                                  self.npu_precision.get(ApiPrecisionCompareColumn.MAX_REL_ERR),
                                                    self.gpu_precision.get(ApiPrecisionCompareColumn.MAX_REL_ERR), 10000.0)
        self.compare_message += max_rel_message
        self.mean_rel_err_ratio, mean_rel_inf_nan_consistency, mean_rel_message = self._calc_ratio(ApiPrecisionCompareColumn.MEAN_REL_ERR,
                                                   self.npu_precision.get(ApiPrecisionCompareColumn.MEAN_REL_ERR),
                                                    self.gpu_precision.get(ApiPrecisionCompareColumn.MEAN_REL_ERR), 10000.0)
        self.compare_message += mean_rel_message
        self.eb_ratio, eb_inf_nan_consistency, eb_message = self._calc_ratio(ApiPrecisionCompareColumn.EB,
                                         self.npu_precision.get(ApiPrecisionCompareColumn.EB),
                                            self.gpu_precision.get(ApiPrecisionCompareColumn.EB), 10000.0)
        self.compare_message += eb_message
        return small_value_inf_nan_consistency, rmse_inf_nan_consistency, max_rel_inf_nan_consistency, mean_rel_inf_nan_consistency, eb_inf_nan_consistency

    def to_column_value(self):
        return [self.small_value_err_ratio, self.small_value_err_status, self.rmse_ratio, 
        self.rmse_status, self.max_rel_err_ratio, self.max_rel_err_status, self.mean_rel_err_ratio, 
        self.mean_rel_err_status, self.eb_ratio, self.eb_status]

    @staticmethod
    def _get_status(ratio, algorithm):
        if math.isnan(ratio) or math.isinf(ratio):
            return CompareConst.PASS
        error_threshold = benchmark_algorithms_thresholds.get(algorithm).get('error_threshold')
        warning_threshold = benchmark_algorithms_thresholds.get(algorithm).get('warning_threshold')
        if ratio > error_threshold:
            return CompareConst.ERROR
        elif ratio > warning_threshold:
            return CompareConst.WARNING
        return CompareConst.PASS


class ULPStandard(Standard):
    def __init__(self, api_name, npu_precision, gpu_precision):
        self.api_name = api_name
        self.npu_precision = npu_precision
        self.gpu_precision = gpu_precision
        self.mean_ulp_err = 0
        self.ulp_err_proportion = 0
        self.ulp_err_proportion_ratio = 1
        self.ulp_err_status = CompareConst.PASS
        self.compare_message = ""

    def __str__(self):
        return f"{self.api_name}"

    def get_result(self):
        self.mean_ulp_err = convert_str_to_float(self.npu_precision.get(ApiPrecisionCompareColumn.MEAN_ULP_ERR))
        gpu_mean_ulp_err = convert_str_to_float(self.gpu_precision.get(ApiPrecisionCompareColumn.MEAN_ULP_ERR))
        inf_nan_consistency = True
        if is_inf_or_nan(self.mean_ulp_err) or is_inf_or_nan(gpu_mean_ulp_err):
            if math.isnan(self.mean_ulp_err) or math.isnan(gpu_mean_ulp_err):
                if math.isnan(self.mean_ulp_err) and math.isnan(gpu_mean_ulp_err):
                    inf_nan_consistency = True
                    self.compare_message += f"{ApiPrecisionCompareColumn.MEAN_ULP_ERR}同为同号inf或nan\n"
                else:
                    inf_nan_consistency = False
                    self.compare_message += f"{ApiPrecisionCompareColumn.MEAN_ULP_ERR}inf或nan不一致\n"
            else:
                if self.mean_ulp_err == gpu_mean_ulp_err:
                    inf_nan_consistency = True
                    self.compare_message += f"{ApiPrecisionCompareColumn.MEAN_ULP_ERR}同为同号inf或nan\n"
                else:
                    inf_nan_consistency = False
                    self.compare_message += f"{ApiPrecisionCompareColumn.MEAN_ULP_ERR}inf或nan不一致\n"
        self.ulp_err_proportion = convert_str_to_float(self.npu_precision.get(ApiPrecisionCompareColumn.ULP_ERR_PROPORTION))
        self.ulp_err_proportion_ratio, ulp_inf_nan_consistency, message = self._calc_ratio(ApiPrecisionCompareColumn.ULP_ERR_PROPORTION,
                                                            self.npu_precision.get(ApiPrecisionCompareColumn.ULP_ERR_PROPORTION),
                                                            self.gpu_precision.get(ApiPrecisionCompareColumn.ULP_ERR_PROPORTION), 10000.0)
        inf_nan_consistency = inf_nan_consistency and ulp_inf_nan_consistency
        self.compare_message += message
        if inf_nan_consistency:
            self.ulp_err_status = self.get_ulp_status(self.npu_precision.get(ApiPrecisionCompareColumn.DEVICE_DTYPE))
        else:
            self.ulp_err_status = CompareConst.ERROR
    
    def get_ulp_status(self, dtype):
        if dtype == paddle.float32:
            if self.mean_ulp_err < 64:
                return CompareConst.PASS
            elif self.ulp_err_proportion < 0.05:
                return CompareConst.PASS
            elif self.ulp_err_proportion_ratio < 1:
                return CompareConst.PASS
            else:
                self.compare_message += "ERROR: ULP误差不满足标准\n"
                return CompareConst.ERROR
        else:
            if self.ulp_err_proportion < 0.001:
                return CompareConst.PASS
            elif self.ulp_err_proportion_ratio < 1:
                return CompareConst.PASS
            else:
                self.compare_message += "ERROR: ULP误差不满足标准\n"
                return CompareConst.ERROR


def write_detail_csv(content, save_path):
    rows = []
    content = ["{:.{}f}".format(item, PRECISION) \
        if isinstance(item, float) else item for item in content]
    rows.append(content)
    write_csv(rows, save_path)


def api_precision_compare(config):
    print_info_log("Start compare task")
    print_info_log(f"Compare task result will be saved in {config.result_csv_path}")
    print_info_log(f"Compare task detail will be saved in {config.details_csv_path}")
    try:
        npu_data = pd.read_csv(config.npu_csv_path)
    except Exception as err:
        print_error_log(f"Open npu csv Error: %s" % str(err))
    check_csv_columns(npu_data.columns, "npu_csv")
    try:
        gpu_data = pd.read_csv(config.gpu_csv_path)
    except Exception as err:
        print_error_log(f"Open gpu csv Error: %s" % str(err))
    check_csv_columns(gpu_data.columns, "gpu_csv")
    detail_csv_title = [ApiPrecisionCompareColumn.get_detail_csv_title()]
    result_csv_title = [ApiPrecisionCompareColumn.get_result_csv_title()]
    write_csv(result_csv_title, config.result_csv_path)
    write_csv(detail_csv_title, config.details_csv_path)
    try:
        analyse_csv(npu_data, gpu_data, config)
    except Exception as err:
        print_error_log(f"Analyse csv Error: %s" % str(err))
    change_mode(config.result_csv_path, FileCheckConst.DATA_FILE_AUTHORITY)
    change_mode(config.details_csv_path, FileCheckConst.DATA_FILE_AUTHORITY)


def analyse_csv(npu_data, gpu_data, config):
    forward_status, backward_status = [], []
    last_api_name, last_api_dtype = None, None
    for _, row_npu in npu_data.iterrows():
        message = ''
        compare_column = ApiPrecisionOutputColumn()
        full_api_name_with_direction_status = row_npu[ApiPrecisionCompareColumn.API_NAME]
        row_gpu = gpu_data[gpu_data[ApiPrecisionCompareColumn.API_NAME] == full_api_name_with_direction_status]
        full_api_name, direction_status, _, _ = full_api_name_with_direction_status.split(".")
        if row_gpu.empty:
            print_warn_log(f'This API : {full_api_name_with_direction_status} does not exist in the GPU data.')
            continue
        if len(row_gpu) > 1:
            msg = f'This API : {full_api_name_with_direction_status} has multiple records in the GPU data.'
            raise CompareException(CompareException.INVALID_DATA_ERROR, msg)
        row_gpu = row_gpu.iloc[0]
        new_status = CompareConst.SPACE
        _, api_name, _ = full_api_name.split("*")
        #当前API的输出为空（例如反向过程中requires_grad=False）,跳过比对
        if row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE].isspace():
            compare_column.api_name = full_api_name_with_direction_status
            compare_column.compare_result = CompareConst.SKIP
            compare_column.compare_message = row_npu[ApiPrecisionCompareColumn.MESSAGE]
            new_status = CompareConst.SKIP
            write_detail_csv(compare_column.to_column_value(), config.details_csv_path)
        else:
            compare_column.api_name = full_api_name_with_direction_status
            if api_name in ThousandthStandardApi:
                new_status = record_thousandth_threshold_result(compare_column, row_npu)
            elif row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] not in BINARY_COMPARE_UNSUPPORT_LIST or api_name in BinaryStandardApi:
                new_status = record_binary_consistency_result(api_name, compare_column, row_npu)                            
            elif api_name in AbsoluteStandardApi:
                new_status = record_absolute_threshold_result(compare_column, row_npu)
            elif api_name in ULPStandardApi and row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] in ULP_COMPARE_SUPPORT_LIST:
                us = ULPStandard(full_api_name_with_direction_status, row_npu, row_gpu)
                new_status = record_ulp_compare_result(compare_column, us)
            elif row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] in BENCHMARK_COMPARE_SUPPORT_LIST:
                bs = BenchmarkStandard(full_api_name_with_direction_status, row_npu, row_gpu)
                new_status = record_benchmark_compare_result(compare_column, bs)
            write_detail_csv(compare_column.to_column_value(), config.details_csv_path)

        if last_api_name is not None and api_name != last_api_name:
            if last_api_dtype in API_PRECISION_COMPARE_UNSUPPORT_LIST:
                message = unsupported_message
                write_csv([[last_api_name, "skip", "skip", message]], config.result_csv_path)
                forward_status, backward_status = [], []
                message = ''
            else:
                forward_result = get_api_checker_result(forward_status)
                backward_result = get_api_checker_result(backward_status)
                message += CompareMessage.get(last_api_name, "") if forward_result == CompareConst.ERROR else ""
                write_csv([[last_api_name, forward_result, backward_result, message]], config.result_csv_path)
                forward_status, backward_status = [], []
                message = ''
                
        is_supported = row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] not in API_PRECISION_COMPARE_UNSUPPORT_LIST
        last_api_name = api_name
        
        last_api_dtype = row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE]
        if not is_supported:
            continue

        if direction_status == 'forward':
            forward_status.append(new_status)
        elif direction_status == 'backward':
            backward_status.append(new_status)
        else:
            print_error_log(f"Invalid direction status: {direction_status}")

    if last_api_name is not None:
        if last_api_dtype in API_PRECISION_COMPARE_UNSUPPORT_LIST:
            message = unsupported_message
            write_csv([[last_api_name, "skip", "skip", message]], config.result_csv_path)
        else:
            forward_result = get_api_checker_result(forward_status)
            backward_result = get_api_checker_result(backward_status)
            message += CompareMessage.get(last_api_name, "") if forward_result == CompareConst.ERROR else ""
            write_csv([[last_api_name, forward_result, backward_result, message]], config.result_csv_path)


def check_error_rate(npu_error_rate):
    return CompareConst.PASS if convert_str_to_float(npu_error_rate) == 0 else CompareConst.ERROR


def get_absolute_threshold_result(row_npu):
    inf_nan_error_ratio = convert_str_to_float(row_npu[ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO])
    rel_err_ratio = convert_str_to_float(row_npu[ApiPrecisionCompareColumn.REL_ERR_RATIO])
    abs_err_ratio = convert_str_to_float(row_npu[ApiPrecisionCompareColumn.ABS_ERR_RATIO])

    inf_nan_result = CompareConst.PASS if inf_nan_error_ratio == 0 else CompareConst.ERROR
    rel_err_result = CompareConst.PASS if rel_err_ratio == 0 else CompareConst.ERROR
    abs_err_result = CompareConst.PASS if abs_err_ratio == 0 else CompareConst.ERROR

    if CompareConst.ERROR in [inf_nan_result, rel_err_result, abs_err_result]:
        absolute_threshold_result = CompareConst.ERROR
    else:
        absolute_threshold_result = CompareConst.PASS

    return {
        "inf_nan_error_ratio": inf_nan_error_ratio,
        "inf_nan_result": inf_nan_result,
        "rel_err_ratio": rel_err_ratio,
        "rel_err_result": rel_err_result,
        "abs_err_ratio": abs_err_ratio,
        "abs_err_result": abs_err_result,
        "absolute_threshold_result": absolute_threshold_result,
    }


def get_api_checker_result(status):
    if not status:
        return CompareConst.SPACE
    if all(item == CompareConst.SKIP for item in status):
        return CompareConst.SKIP
    for const in (CompareConst.ERROR, CompareConst.WARNING):
        if const in status:
            return const
    return CompareConst.PASS


def check_csv_columns(columns, csv_type):
    required_columns = ApiPrecisionCompareColumn.to_required_columns()
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        msg = f"The following columns {','.join(missing_columns)} are missing in{csv_type}"
        raise CompareException(CompareException.INVALID_DATA_ERROR, msg)


def record_binary_consistency_result(api_name, compare_column, row_npu):
    new_status = check_error_rate(row_npu[ApiPrecisionCompareColumn.ERROR_RATE])
    compare_column.error_rate = row_npu[ApiPrecisionCompareColumn.ERROR_RATE]
    compare_column.error_rate_status = new_status
    compare_column.compare_result = new_status
    compare_column.compare_algorithm = "二进制一致法"
    message = ''
    if compare_column.error_rate_status == CompareConst.ERROR:
        message += "ERROR: 二进制一致错误率超过阈值\n"
        message += CompareMessage.get(api_name, "")
    compare_column.compare_message = message
    return new_status


def record_absolute_threshold_result(compare_column, row_npu):
    absolute_threshold_result = get_absolute_threshold_result(row_npu)
    compare_column.inf_nan_error_ratio = absolute_threshold_result.get("inf_nan_error_ratio")
    compare_column.inf_nan_error_ratio_status = absolute_threshold_result.get("inf_nan_result")
    compare_column.rel_err_ratio = absolute_threshold_result.get("rel_err_ratio")
    compare_column.rel_err_ratio_status = absolute_threshold_result.get("rel_err_result")
    compare_column.abs_err_ratio = absolute_threshold_result.get("abs_err_ratio")
    compare_column.abs_err_ratio_status = absolute_threshold_result.get("abs_err_result")
    compare_column.compare_result = absolute_threshold_result.get("absolute_threshold_result")
    compare_column.compare_algorithm = "绝对阈值法"
    message = ''
    if compare_column.inf_nan_error_ratio_status == CompareConst.ERROR:
        message += "ERROR: inf/nan错误率超过阈值\n"
    if compare_column.rel_err_ratio_status == CompareConst.ERROR:
        message += "ERROR: 相对误差错误率超过阈值\n"
    if compare_column.abs_err_ratio_status == CompareConst.ERROR:
        message += "ERROR: 绝对误差错误率超过阈值\n"
    compare_column.compare_message = message
    return compare_column.compare_result


def record_benchmark_compare_result(compare_column, bs):
    bs.get_result()
    compare_column.small_value_err_ratio = bs.small_value_err_ratio
    compare_column.small_value_err_status = bs.small_value_err_status
    compare_column.rmse_ratio = bs.rmse_ratio
    compare_column.rmse_status = bs.rmse_status
    compare_column.max_rel_err_ratio = bs.max_rel_err_ratio
    compare_column.max_rel_err_status = bs.max_rel_err_status
    compare_column.mean_rel_err_ratio = bs.mean_rel_err_ratio
    compare_column.mean_rel_err_status = bs.mean_rel_err_status
    compare_column.eb_ratio = bs.eb_ratio
    compare_column.eb_status = bs.eb_status
    compare_column.compare_result = bs.final_result
    compare_column.compare_algorithm = "标杆比对法"
    compare_column.compare_message = bs.compare_message
    for status_attr, messages in benchmark_message.items():
        status_value = getattr(compare_column, status_attr)
        if status_value in messages:
            compare_column.compare_message += messages[status_value]
    return compare_column.compare_result


def record_ulp_compare_result(compare_column, us):
    us.get_result()
    compare_column.mean_ulp_err = us.mean_ulp_err
    compare_column.ulp_err_proportion = us.ulp_err_proportion
    compare_column.ulp_err_proportion_ratio = us.ulp_err_proportion_ratio
    compare_column.ulp_err_status = us.ulp_err_status
    compare_column.compare_result = us.ulp_err_status
    compare_column.compare_algorithm = "ULP误差比对法"
    compare_column.compare_message = us.compare_message
    return compare_column.compare_result


def check_thousandth_rate(thousandth_rate):
    return CompareConst.PASS if convert_str_to_float(thousandth_rate) >= 0.999 else CompareConst.ERROR


def record_thousandth_threshold_result(compare_column, row_npu):
    new_status = check_thousandth_rate(row_npu[ApiPrecisionCompareColumn.REL_ERR_THOUSANDTH])
    compare_column.rel_err_thousandth = row_npu[ApiPrecisionCompareColumn.REL_ERR_THOUSANDTH]
    compare_column.rel_err_thousandth_status = new_status
    compare_column.compare_result = new_status
    compare_column.compare_algorithm = "双千指标法"
    message = ''
    if compare_column.rel_err_thousandth_status == CompareConst.ERROR:
        message += "ERROR: 双千指标不达标\n"
    compare_column.compare_message = message
    return compare_column.compare_result


def _api_precision_compare(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    _api_precision_compare_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    _api_precision_compare_command(args)


def _api_precision_compare_command(args):
    npu_csv_path = get_validated_result_csv_path(args.npu_csv_path, 'detail')
    gpu_csv_path = get_validated_result_csv_path(args.gpu_csv_path, 'detail')
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    check_path_before_create(out_path)
    create_directory(out_path)
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    result_csv_path = os.path.join(out_path, API_PRECISION_COMPARE_RESULT_FILE_NAME)
    details_csv_path = os.path.join(out_path, API_PRECISION_COMPARE_DETAILS_FILE_NAME)
    compare_config = CompareConfig(npu_csv_path, gpu_csv_path, result_csv_path, details_csv_path)
    api_precision_compare(compare_config)


def _api_precision_compare_parser(parser):
    parser.add_argument("-npu", "--detail1", dest="npu_csv_path", default="", type=str,
                        help="<Required> , Accuracy_checking_details.csv generated on the NPU by using the "
                             "api_accuracy_checker tool.",
                        required=True)
    parser.add_argument("-gpu", "--detail2", dest="gpu_csv_path", default="", type=str,
                        help="<Required> Accuracy_checking_details.csv generated on the GPU by using the "
                             "api_accuracy_checker tool.",
                        required=False)
    parser.add_argument("-o", "--output_path", dest="out_path", default="", type=str,
                        help="<optional> The api precision compare task result out path.",
                        required=False)


if __name__ == '__main__':
    _api_precision_compare()
    print_info_log("Compare task completed.")
    