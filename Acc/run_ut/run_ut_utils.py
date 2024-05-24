hf_32_standard_api = ["conv1d", "conv2d"]


class Backward_Message:
    MULTIPLE_BACKWARD_MESSAGE = "Multiple backward is not supported."
    UNSUPPORT_BACKWARD_MESSAGE = "function with out=... arguments don't support automatic differentiation, skip backward."
    NO_BACKWARD_RESULT_MESSAGE = "function backward result is None, skip backward."