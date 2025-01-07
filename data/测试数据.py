import numpy as np
import glob

from torch_geometric.io.obj import yield_file

xiaoceng = "6_1"

# 假设你的文件以某种格式命名，比如*.txt
train_file_list = glob.glob(fr'C:\工作数据\KCN数据\{xiaoceng}\训练/*.txt')
test_file_list = glob.glob(fr'C:\工作数据\KCN数据\{xiaoceng}\测试/*.txt')

# 用字典来存储所有文件的坐标
coordinates_set = None

# 用字典来存储每个文件的第三列数据
third_columns_dict = {}

# 遍历每个文件，提取坐标和第三列数据
for file in train_file_list:
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
    for file in train_file_list:
        if coord in third_columns_dict[file]:
            row.append(third_columns_dict[file].get(coord, np.nan))  # 如果找不到该坐标，填充为 NaN
    merged_data.append(row)

# 将最终数据转换为 NumPy 数组
merged_data = np.array(merged_data)

X_train = np.hstack((merged_data[:, :2], merged_data[:, 3:]))
y_train = merged_data[:, 2]

cols_to_normalize = X_train[:, 2:]  # 从第三列到最后一列
min_vals = np.min(cols_to_normalize, axis=0)
max_vals = np.max(cols_to_normalize, axis=0)

normalized_cols = (cols_to_normalize - min_vals) / (max_vals - min_vals)
X_train[:, 2:] = normalized_cols

# 用字典来存储所有文件的坐标
coordinates_set = None

# 用字典来存储每个文件的第三列数据
third_columns_dict = {}

# 遍历每个文件，提取坐标和第三列数据
for file in test_file_list:
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
    for file in test_file_list:
        if coord in third_columns_dict[file]:
            row.append(third_columns_dict[file].get(coord, np.nan))  # 如果找不到该坐标，填充为 NaN
    merged_data.append(row)

# 将最终数据转换为 NumPy 数组
merged_data = np.array(merged_data)

X_test = np.hstack((merged_data[:, :2], merged_data[:, 3:]))
y_test = merged_data[:, 2]

cols_to_normalize = X_test[:, 2:]
min_vals = np.min(cols_to_normalize, axis=0)
max_vals = np.max(cols_to_normalize, axis=0)
normalized_cols = (cols_to_normalize - min_vals) / (max_vals - min_vals)
X_test[:, 2:] = normalized_cols

np.savez(fr'C:\工作数据\KCN数据\{xiaoceng}\data.npz', Xtrain=X_train, Ytrain=y_train, Xtest=X_test, Ytest=y_test)



# np.savetxt(r'C:\工作数据\KCN数据\6_2\测试\test_X.npz', merged_data, fmt='%.6f')

print(fr"合并完成，结果已保存到 C:\工作数据\KCN数据\{xiaoceng}\data.npz")
