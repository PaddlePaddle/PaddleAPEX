"""
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
"""
from compare_utils.compare_utils import CompareConst


class CompareColumn:
    def __init__(self):
        self.bench_type = CompareConst.SPACE
        self.device_type = CompareConst.SPACE
        self.shape = CompareConst.SPACE
        self.cosine_sim = CompareConst.SPACE
        self.max_abs_err = CompareConst.SPACE
        self.rel_err_hundredth = CompareConst.SPACE
        self.rel_err_thousandth = CompareConst.SPACE
        self.rel_err_ten_thousandth = CompareConst.SPACE
        self.error_rate = CompareConst.SPACE
        self.EB = CompareConst.SPACE
        self.RMSE = CompareConst.SPACE
        self.small_value_err_ratio = CompareConst.SPACE
        self.Max_rel_error = CompareConst.SPACE
        self.Mean_rel_error = CompareConst.SPACE
        self.inf_nan_error_ratio = CompareConst.SPACE
        self.rel_err_ratio = CompareConst.SPACE
        self.abs_err_ratio = CompareConst.SPACE

    def to_column_value(self, is_pass, message):
        return [
            self.bench_type,
            self.device_type,
            self.shape,
            self.cosine_sim,
            self.max_abs_err,
            self.rel_err_hundredth,
            self.rel_err_thousandth,
            self.rel_err_ten_thousandth,
            self.error_rate,
            self.EB,
            self.RMSE,
            self.small_value_err_ratio,
            self.Max_rel_error,
            self.Mean_rel_error,
            self.inf_nan_error_ratio,
            self.rel_err_ratio,
            self.abs_err_ratio,
            is_pass,
            message,
        ]


class ApiPrecisionOutputColumn:
    def __init__(self):
        self.api_name = CompareConst.SPACE
        self.small_value_err_ratio = CompareConst.SPACE
        self.small_value_err_status = CompareConst.SPACE
        self.rmse_ratio = CompareConst.SPACE
        self.rmse_status = CompareConst.SPACE
        self.max_rel_err_ratio = CompareConst.SPACE
        self.max_rel_err_status = CompareConst.SPACE
        self.mean_rel_err_ratio = CompareConst.SPACE
        self.mean_rel_err_status = CompareConst.SPACE
        self.eb_ratio = CompareConst.SPACE
        self.eb_status = CompareConst.SPACE
        self.inf_nan_error_ratio = CompareConst.SPACE
        self.inf_nan_error_ratio_status = CompareConst.SPACE
        self.rel_err_ratio = CompareConst.SPACE
        self.rel_err_ratio_status = CompareConst.SPACE
        self.abs_err_ratio = CompareConst.SPACE
        self.abs_err_ratio_status = CompareConst.SPACE
        self.error_rate = CompareConst.SPACE
        self.error_rate_status = CompareConst.SPACE
        self.hundred_percent_ratio = CompareConst.SPACE
        self.thounsand_percent_ratio = CompareConst.SPACE
        self.million_percent_ratio = CompareConst.SPACE
        self.compare_result = CompareConst.SPACE
        self.compare_algorithm = CompareConst.SPACE
        self.compare_message = CompareConst.SPACE

    def to_column_value(self):
        return [
            self.api_name,
            self.small_value_err_ratio,
            self.small_value_err_status,
            self.rmse_ratio,
            self.rmse_status,
            self.max_rel_err_ratio,
            self.max_rel_err_status,
            self.mean_rel_err_ratio,
            self.mean_rel_err_status,
            self.eb_ratio,
            self.eb_status,
            self.inf_nan_error_ratio,
            self.inf_nan_error_ratio_status,
            self.rel_err_ratio,
            self.rel_err_ratio_status,
            self.abs_err_ratio,
            self.abs_err_ratio_status,
            self.error_rate,
            self.error_rate_status,
            self.hundred_percent_ratio,
            self.thounsand_percent_ratio,
            self.million_percent_ratio,
            self.compare_result,
            self.compare_algorithm,
            self.compare_message,
        ]
