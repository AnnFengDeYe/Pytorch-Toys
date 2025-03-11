import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import time  # 导入 time 模块


# 1. 数据预处理

# 配置数据下载相关参数
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('.', 'data')):
    """
    下载一个DATA_HUB中的文件，返回本地文件名。
    """
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            print(f"使用缓存文件: {fname}")
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """
    下载并解压zip/tar文件。
    """
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    print(f"解压文件到: {base_dir}")
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """
    下载DATA_HUB中的所有文件。
    """
    for name in DATA_HUB:
        download(name)

# 加载数据
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

print(f"训练数据形状: {train_data.shape}")
print(f"测试数据形状: {test_data.shape}")
print(f"训练数据前4行，部分列:\n{train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]}")

# 数据预处理：合并、特征工程、标准化、独热编码
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(f"合并后特征数据最后4行:\n{all_features.iloc[-4:-1, :]}")

# 特征工程：添加总面积特征
all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)
print(f"独热编码后特征数据形状: {all_features.shape}")


# 转换为 PyTorch 张量
n_train = train_data.shape[0]
print(f"训练集样本数: {n_train}")
print(f"特征数据类型:\n{all_features.dtypes}")

# 2. 设置神经网络模型结构, 并移动到设备
class Config:
    def __init__(self):
        self.k = 5
        self.num_epochs = 100
        self.lr = 0.01  # 调小学习率
        self.weight_decay = 0.1  # 增大权重衰减
        self.batch_size = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.hidden_units1 = 256  # 第一个隐藏层单元数
        self.hidden_units2 = 128  # 第二个隐藏层单元数

config = Config() # 初始化配置

def get_net(input_features):
    """
    创建一个包含两个隐藏层的神经网络模型。
    """
    net = nn.Sequential(
        nn.Linear(input_features, config.hidden_units1),
        nn.ReLU(),
        nn.Dropout(0.1),  # 添加 Dropout
        nn.Linear(config.hidden_units1, config.hidden_units2),
        nn.ReLU(),
        nn.Dropout(0.1),  # 添加 Dropout
        nn.Linear(config.hidden_units2, 1)
    )
    return net

# 损失函数
loss = nn.MSELoss()


# 3. 训练相关的函数
def log_rmse(net, features, labels):
    """
    计算对数均方根误差 (log RMSE)。
    """
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, config):
    """
    训练模型。
    """
    train_ls, test_ls = [], []
    train_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_features, train_labels), config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    net = net.to(config.device) # 将模型移动到设备

    start_time = time.time() # 记录开始时间

    for epoch in range(config.num_epochs):
        for X, y in train_iter:
            X, y = X.to(config.device), y.to(config.device)  # 将数据移动到设备
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_features, train_labels = train_features.to(config.device), train_labels.to(config.device)  # 移动到设备

        train_rmse = log_rmse(net, train_features, train_labels)
        train_ls.append(train_rmse)

        if test_labels is not None:
            test_features, test_labels = test_features.to(config.device), test_labels.to(config.device) #移动到设备
            test_rmse = log_rmse(net, test_features, test_labels)
            test_ls.append(test_rmse)
            print(f'Epoch {epoch+1}, Train RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}')
        else:
            print(f'Epoch {epoch+1}, Train RMSE: {train_rmse:.6f}')

    end_time = time.time() # 记录结束时间
    total_time = end_time - start_time # 计算总时间
    print(f"总训练时间: {total_time:.2f} 秒")

    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    """
    获取 K 折交叉验证的训练集和验证集。
    """
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, config):
    """
    执行 K 折交叉验证。
    """
    train_l_sum, valid_l_sum = 0, 0

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, config)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plt.figure()
            plt.plot(list(range(1, config.num_epochs + 1)), train_ls, label='train')
            plt.plot(list(range(1, config.num_epochs + 1)), valid_ls, label='valid')
            plt.xlabel('epoch')
            plt.ylabel('rmse')
            plt.xlim([1, config.num_epochs])
            plt.yscale('log')
            plt.legend()
            plt.show()

        print(f'折{i + 1}，训练log rmse: {float(train_ls[-1]):f}, '
              f'验证log rmse: {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


# 4. 训练结果

def train_and_pred(train_features, test_features, train_labels, test_data, config):
    """
    在整个训练集上训练模型，并在测试集上进行预测。
    """
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None, config)

    plt.figure()
    plt.plot(np.arange(1, config.num_epochs + 1), train_ls, label='train')
    plt.xlabel('epoch')
    plt.ylabel('log rmse')
    plt.xlim([1, config.num_epochs])
    plt.yscale('log')
    plt.legend()
    plt.show()

    print(f'训练log rmse：{float(train_ls[-1]):f}')
    test_features = test_features.to(config.device)  # 将测试特征移动到设备
    preds = net(test_features).detach().cpu().numpy() # 将预测结果移回CPU
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
    print("预测结果已保存到 submission.csv")


train_features = torch.tensor(all_features[:n_train].values.astype(np.float32), dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values.astype(np.float32), dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# K 折交叉验证 或 训练和预测 (选择一个运行)
# K折交叉验证
train_l, valid_l = k_fold(config.k, train_features, train_labels, config)
print(f'{config.k}-折验证: 平均训练log rmse: {float(train_l):f}, 平均验证log rmse: {float(valid_l):f}')


# 预测
print(f"当前工作目录: {os.getcwd()}")

train_and_pred(train_features, test_features, train_labels, test_data, config)