import pandas as pd
import glob
import os

# 定义包含 CSV 文件的目录
csv_dir = 'log/'

# 使用 glob 模块查找目录中所有的 CSV 文件
csv_files = glob.glob(os.path.join(csv_dir, '*forward*.csv'))
dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

combined_df = pd.concat(dataframes, axis=0, ignore_index=True)

# 假设所有 CSV 的列名和顺序相同，按第二列排序
# 使用 iloc[:, 1] 获取第二列的列名
second_column_name = combined_df.columns[1]
# 按第二列排序
sorted_df = combined_df.sort_values(by=second_column_name)
# 输出排序后的 DataFrame
print(sorted_df)
# 可选：将排序后的 DataFrame 保存为新的 CSV 文件
sorted_df.to_csv('sorted_combined_forward.csv', index=False)


# 使用 glob 模块查找目录中所有的 CSV 文件
csv_files = glob.glob(os.path.join(csv_dir, '*backward*.csv'))
dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

combined_df = pd.concat(dataframes, axis=0, ignore_index=True)

# 假设所有 CSV 的列名和顺序相同，按第二列排序
# 使用 iloc[:, 1] 获取第二列的列名
second_column_name = combined_df.columns[1]
# 按第二列排序
sorted_df = combined_df.sort_values(by=second_column_name)
# 输出排序后的 DataFrame
print(sorted_df)
# 可选：将排序后的 DataFrame 保存为新的 CSV 文件
sorted_df.to_csv('sorted_combined_backward.csv', index=False)
