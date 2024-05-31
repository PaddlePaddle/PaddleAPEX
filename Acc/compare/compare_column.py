from compare.compare_utils import CompareConst


class CompareColumn:
    def __init__(self):
        self.bench_type = CompareConst.SPACE
        self.npu_type = CompareConst.SPACE
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
        return [self.bench_type, self.npu_type, self.shape, self.cosine_sim, self.max_abs_err, self.rel_err_hundredth,
                self.rel_err_thousandth, self.rel_err_ten_thousandth, self.error_rate, self.EB, self.RMSE, 
                self.small_value_err_ratio, self.Max_rel_error, self.Mean_rel_error, self.inf_nan_error_ratio, 
                self.rel_err_ratio, self.abs_err_ratio, is_pass, message]


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
        self.compare_result = CompareConst.SPACE
        self.compare_algorithm = CompareConst.SPACE
        self.compare_message = CompareConst.SPACE

    def to_column_value(self):
        return [self.api_name, self.small_value_err_ratio, self.small_value_err_status, self.rmse_ratio, 
                self.rmse_status, self.max_rel_err_ratio, self.max_rel_err_status, self.mean_rel_err_ratio, 
                self.mean_rel_err_status, self.eb_ratio, self.eb_status, self.inf_nan_error_ratio, 
                self.inf_nan_error_ratio_status, self.rel_err_ratio, self.rel_err_ratio_status, self.abs_err_ratio, 
                self.abs_err_ratio_status, self.error_rate, self.error_rate_status, self.compare_result, 
                self.compare_algorithm, self.compare_message]
        