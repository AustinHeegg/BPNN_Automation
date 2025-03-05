import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from src.baseHJC import readColumn


# 读取并修改column内容
def getAllFfmData(files, selected_columns, rowSkip=1):
    data_lists = [[] for _ in selected_columns]

    for f in files:
        for idx, col in enumerate(selected_columns):
            data = readColumn(f, col, rowSkip=rowSkip)
            data_lists[idx].extend(data)

    data_arrays = [np.array(data) for data in data_lists]
    return data_arrays


def prepare_data(data_files, batch_size, test_size, random_state, selected_input_columns, selected_output_columns,
                 rowSkip):
    data_arrays = getAllFfmData(data_files, selected_input_columns + selected_output_columns, rowSkip)

    # 将训练和测试数据转换为张量
    X = torch.tensor(np.column_stack([data_arrays[i] for i in selected_input_columns]).astype(np.float32))
    y = torch.tensor(np.column_stack([data_arrays[i] for i in selected_output_columns]).astype(np.float32))

    # np.set_printoptions(precision=15, floatmode="fixed")
    # print("Input data (X_test):")
    # for i in range(X.size(0)):  # 遍历行
    #     inputs = (X[i, 0].item(), X[i, 1].item())  # 获取每行的两个输入
    #     print(f"({inputs[0]:.15f}, {inputs[1]:.15f})")  # 格式化输出
    #
    # print("Output data (y_test):")
    # for i in range(y.size(0)):  # 遍历行
    #     outputs = (y[i, 0].item(), y[i, 1].item())  # 获取每行的两个输出
    #     print(f"({outputs[0]:.15f}, {outputs[1]:.15f})")  # 格式化输出


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 将数据转换为 PyTorch 张量
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    else:
        X_train = X_train.clone().detach()

    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float32)
    else:
        y_train = y_train.clone().detach()

    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    else:
        X_test = X_test.clone().detach()

    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.float32)
    else:
        y_test = y_test.clone().detach()

    # 创建创建数据加载器
    train_dataset = data.TensorDataset(X_train, y_train)
    test_dataset = data.TensorDataset(X_test, y_test)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, X_test, y_test
