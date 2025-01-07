import numpy as np
import glob

xiaoceng = "6_1"

file_list = glob.glob(fr'C:\工作数据\KCN数据\{xiaoceng}\分频与反演属性/*.txt')

# 用字典来存储所有文件的坐标
coordinates_set = None

# 用字典来存储每个文件的第三列数据
third_columns_dict = {}

# 遍历每个文件，提取坐标和第三列数据
for file in file_list:
    data = np.loadtxt(file, delimiter=' ', usecols=(0, 1, 2))
    coordinates = set(tuple(row[:2]) for row in data)  # 提取坐标并转换为集合

    # 如果是第一个文件，初始化坐标集合
    if coordinates_set is None:
        coordinates_set = coordinates
    else:
        # 获取所有文件中共有的坐标
        coordinates_set &= coordinates

    # 将每个文件的第三列数据保存到字典
    third_columns_dict[file] = {tuple(row[:2]): row[2] for row in data}

# 现在，coordinates_set 包含所有文件中共有的坐标
# 初始化最终合并的结果数组（前两列相同的坐标）
merged_data = []

# 将所有公共坐标行组合起来
for coord in coordinates_set:
    row = [coord[0], coord[1]]  # 前两列是坐标
    # 获取每个文件中对应坐标的第三列数据
    for file in file_list:
        if coord in third_columns_dict[file]:
            row.append(third_columns_dict[file].get(coord, np.nan))  # 如果找不到该坐标，填充为 NaN
    merged_data.append(row)

# 将最终数据转换为 NumPy 数组
merged_data = np.array(merged_data)

# 1. 隔行抽稀（按原始顺序）
merged_data_sparse = merged_data[::2]  # 取每隔一行的样本

# 2. 按第一列排序
merged_data_sparse = merged_data_sparse[merged_data_sparse[:, 0].argsort()]

# 3. 再次隔行抽稀（按排序后的数据）
final_sparse_data = merged_data_sparse[::2]  # 取每隔一行的样本

X = final_sparse_data
cols_to_normalize = X[:, 2:]  # 从第三列到最后一列
min_vals = np.min(cols_to_normalize, axis=0)
max_vals = np.max(cols_to_normalize, axis=0)

normalized_cols = (cols_to_normalize - min_vals) / (max_vals - min_vals)
X[:, 2:] = normalized_cols

# 5. 保存数据
np.savez(fr'C:\工作数据\KCN数据\{xiaoceng}\predicting_data.npz', X=X)

