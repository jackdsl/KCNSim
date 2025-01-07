import os.path

import torch
import numpy as np
from matplotlib import pyplot as plt
from pyexpat import features
from scipy.stats import stats, pearsonr
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import kcn
import data
from tqdm import tqdm

def run_kcn(args):
    """ Train and test a KCN model on a train-test split  

    Args
    ----
    args : argparse.Namespace object, which contains the following attributes:
        - 'model' : str, which is one of 'gcn', 'gcn_gat', 'gcn_sage'
        - 'n_neighbors' : int, number of neighbors
        - 'hidden1' : int, number of units in hidden layer 1
        - 'dropout' : float, the dropout rate in a dropout layer 
        - 'lr' : float, learning rate of the Adam optimizer
        - 'epochs' : int, number of training epochs
        - 'es_patience' : int, patience for early stopping
        - 'batch_size' : int, batch size
        - 'dataset' : str, path to the data file
        - 'last_activation' : str, activation for the last layer
        - 'weight_decay' : float, weight decay for the Adam optimizer
        - 'length_scale' : float, length scale for RBF kernel
        - 'loss_type' : str, which is one of 'squared_error', 'nll_error'
        - 'validation_size' : int, validation size
        - 'gcn_kriging' : bool, whether to use gcn kriging
        - 'sparse_input' : bool, whether to use sparse matrices
        - 'device' : torch.device, which is either 'cuda' or 'cpu'

    """
    # This function has the following three steps:
    # 1) loading data; 2) spliting the data into training and test subsets; 3) normalizing data 
    if args.dataset == "bird_count":
        trainset, testset = data.load_bird_count_data(args)
    else:
        raise Exception(f"The repo does not support this dataset yet: args.dataset={args.dataset}")

    print(f"The {args.dataset} dataset has {len(trainset)} training instances and {len(testset)} test instances.")

    num_total_train = len(trainset)
    num_valid = args.validation_size
    num_train = num_total_train - args.validation_size

    # initialize a kcn model
    # 1) the entire training set including validation points are recorded by the model and will 
    # be looked up in neighbor searches
    # 2) the model will pre-compute neighbors for a training or validation instance to avoid repeated neighbor search
    # 3) if a data point appears in training set and validation set, its neighbors does not include itself
    model = kcn.KCN(trainset, args)
    model = model.to(args.device)

    loss_func = torch.nn.L1Loss(reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_train_error = []
    epoch_valid_error = []

    # the training loop
    model.train()


    for epoch in range(args.epochs):

        batch_train_error = [] 

        # use training indices directly because it will be used to retrieve pre-computed neighbors
        for i in tqdm(range(0, num_train, args.batch_size)):

            # fetch a batch of data  
            batch_ind = range(i, min(i + args.batch_size, num_train))
            batch_coords, batch_features, batch_y = model.trainset[batch_ind]

            # make predictions and compute the average loss
            pred = model(batch_coords, batch_features, batch_ind)
            loss = loss_func(pred, batch_y.to(args.device))

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record the training error
            batch_train_error.append(loss.item())

        train_error = sum(batch_train_error) / len(batch_train_error)
        epoch_train_error.append(train_error)

        # fetch the validation set
        valid_ind = range(num_train, num_total_train)
        valid_coords, valid_features, valid_y = model.trainset[valid_ind]

        # make predictions and calculate the error
        valid_pred = model(valid_coords, valid_features, valid_ind)
        valid_error = loss_func(valid_pred, valid_y.to(args.device))

        epoch_valid_error.append(valid_error.item())

        print(f"Epoch: {epoch},", f"train error: {train_error},", f"validation error: {valid_error}")

        # # check whether to stop
        # if (epoch > args.es_patience) and \
        #         (np.mean(np.array(epoch_valid_error[-3:])) >
        #          np.mean(np.array(epoch_valid_error[-(args.es_patience + 3):-3]))):
        #     print("\nEarly stopping at epoch {}".format(epoch))
        #     break

    # test the model
    model.eval()

    test_preds = model(testset.coords, testset.features)
    test_y = testset.y.to(args.device)

    test_error = loss_func(test_preds, test_y)
    test_error = torch.mean(test_error).item()

    print(f"Test error is {test_error}")
    r2 = r2_score(test_preds, test_y)
    print(f"R2 score is {r2}")

    # 确保 test_preds 和 test_y 是一维数组
    test_preds_np = test_preds.detach().cpu().numpy().flatten()
    test_y_np = test_y.detach().cpu().numpy().flatten()

    if args.plot:

        # 可视化：柱状图对比 test_preds 和 test_y
        plt.figure(figsize=(10, 5))

        # 获取样本的索引
        x = np.arange(len(test_preds_np))

        # 设置柱宽
        bar_width = 0.35

        # 绘制 test_preds 的柱状图
        plt.bar(x - bar_width / 2, test_preds_np, bar_width, label='Predictions (test_preds)', color='b')

        # 绘制 test_y 的柱状图
        plt.bar(x + bar_width / 2, test_y_np, bar_width, label='Ground Truth (test_y)', color='r')

        # 添加标题和标签
        plt.title('Test Predictions vs Ground Truth')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')

        # 显示图例
        plt.legend()

        # 显示图形
        plt.tight_layout()
        plt.show()

        # 计算皮尔逊相关系数
        corr_coefficient, _ = pearsonr(test_preds_np, test_y_np)

        # 可视化：绘制 test_preds 对 test_y 的二维散点图
        plt.figure(figsize=(8, 6))

        # 绘制散点图
        plt.scatter(test_preds_np, test_y_np, label=f'Predictions vs Ground Truth (R = {corr_coefficient:.2f})',
                    color='b', alpha=0.5)

        # 添加标题和标签
        plt.title('Predictions vs Ground Truth with Correlation Coefficient (R)')
        plt.xlabel('Predictions (test_preds)')
        plt.ylabel('Ground Truth (test_y)')

        # 显示图例
        plt.legend()

        # 显示图形
        plt.tight_layout()
        plt.show()

    if args.predict:
        unknown_X = data.load_predicting_data(args)

        # sand_thickness_preds = model(unknown_X.coords, unknown_X.features)
        # unknown_X.coords = unknown_X.coords.to(args.device)
        #
        # result = torch.cat((unknown_X.coords, sand_thickness_preds), dim=1)
        # # 将结果转换为 NumPy 数组
        # result_np = result.detach().to("cpu").numpy()
        # # 将 NumPy 数组保存到文本文件
        # np.savetxt(args.Spatial_data_predicted, result_np, fmt='%f')  # 'result.txt' 是保存的文件名，'%f' 表示保存浮点数

        # 将 coords 和 features 打包为一个数据集
        dataset = TensorDataset(unknown_X.coords, unknown_X.features)

        # 创建一个 DataLoader，其中 shuffle=True 会自动打乱数据
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        predictions = []
        coords = unknown_X.coords
        features = unknown_X.features
        batch_size = 128
        # 预测过程
        # 按批次进行预测
        with torch.no_grad():  # 禁用梯度计算，节省内存
            for i in tqdm(range(0, len(coords), batch_size), desc="Predicting", ncols=100):
                # 提取当前批次的数据
                batch_coords = coords[i:i + batch_size]
                batch_features = features[i:i + batch_size]

                # 通过模型进行预测
                batch_preds = model(batch_coords, batch_features)

                # 将当前批次的预测结果添加到列表中
                predictions.append(batch_preds)

        # 将所有批次的预测结果拼接起来
        predictions = torch.cat(predictions, dim=0)

        # 拼接坐标和预测结果
        result = torch.cat((coords, predictions.cpu()), dim=1)

        # 将结果转换为 NumPy 数组并保存
        result_np = result.detach().to("cpu").numpy()
        np.savetxt(args.Spatial_data_predicted, result_np, fmt='%f')
def r2_score(y_true, y_pred):
    """
    计算 R²（决定系数）。

    参数:
    - y_true: 真实值（Tensor）
    - y_pred: 预测值（Tensor）

    返回:
    - R²值
    """
    # 计算总平方和（TSS）
    total_variance = torch.sum((y_true - torch.mean(y_true)) ** 2)

    # 计算残差平方和（RSS）
    residual_variance = torch.sum((y_true - y_pred) ** 2)

    # 计算 R²
    r2 = 1 - (residual_variance / total_variance)

    return r2.item()  # 返回 Python 数字

