import os

def createDir(file_name):
    if not os.path.exists(file_name):
        os.makedirs(file_name)


def generate_true_random_number(num_bytes):
    """ 生成真随机数，从 /dev/urandom 读取指定数量的字节 """
    # 从 /dev/urandom 读取 num_bytes 个字节
    random_bytes = os.urandom(num_bytes)
    # 将字节转换为整数
    #print(int.from_bytes(random_bytes, 'big'))
    return int.from_bytes(random_bytes, 'big')


# inf, nan
def get_rounded_num(x, round_up=True):
    if math.isinf(x) or math.isnan(x):
        msg = f"warning, x is inf or nan"
        print(msg, x)
        return x
    if abs(x) <= 1e-10:
        return 0
    
    abs_x = abs(x)
    log_x = math.log10(abs_x)
    round_log_x = math.floor(log_x) if round_up ^ (x > 0) else math.ceil(log_x)
    
    result = 10**round_log_x
    return result if x >= 0 else -result


def generate_random_array(mean, max_value, shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # 首先生成标准正态分布的随机数组
    random_array = np.random.randn(*shape).astype(np.float32)
    # 计算当前随机数组的最大值
    current_max = random_array.max()
    # 计算缩放因子，使得新的最大值为给定的max_value
    scale_factor = max_value / current_max
    # 对数组进行缩放
    random_array *= scale_factor
    # 计算当前数组的均值
    current_mean = random_array.mean()
    # 计算偏移量，使得新的均值为给定的mean
    shift_value = mean - current_mean
    # 对数组进行偏移
    random_array += shift_value
    return random_array


def create_random_tensor(shape, dtype, seed, min_v=-1, mean_v=None, max_v=1):
    if mean_v == None:
        paddle.seed(seed)       
        data = np.random.uniform(min_v, max_v, shape).astype("float32")
    else:
        data = generate_random_array(mean_v, max_v, shape, seed)
    data = paddle.to_tensor(data, stop_gradient=False).cast(dtype)
    return data
