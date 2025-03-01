import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import os  # 导入 os 模块


class Config:
    """配置类，用于存储所有超参数和可配置项"""
    # 数据文件路径
    data_file = 'equips.json'

    # 模型保存路径
    model_path = 'price_prediction_model.pth'
    scaler_path = 'price_scaler.pth'
    encoder_path = 'price_encoder.pth'

    # 训练参数
    epochs = 1000
    learning_rate = 0.001
    batch_size = 128
    weight_decay = 0.01
    dropout_rate = 0.5

    # 模型参数 (这些也可以根据需要调整)
    hidden_layers = [1024, 512, 256, 128, 64] #隐藏层数
    # 辅助属性和主属性的枚举集合
    auxiliary_attributes_options = ["固伤", "法伤", "伤害", "封印", "法暴", "物暴", "狂暴", "穿刺", "法伤结果", "治疗", "速度", "回复", "气血", "防御", "抗封", "抗法暴", "格挡", "法防", "抗物暴"]
    main_attributes_options = ["伤害", "封印命中", "抗封", "法伤", "法防", "防御", "速度"]


    # 自动检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    #如果上述都不支持，就用cpu

class PricePredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(PricePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)  # 更大的第一层
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 128)  # 额外的层
        self.bn4 = nn.BatchNorm1d(128) # 额外的BN层
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)   # 更多的层
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.relu(self.fc4(x)) #更多的层和relu
        x = self.bn4(x)
        x = self.dropout4(x)
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x



def load_and_preprocess_data(config):
    """加载和预处理数据"""
    with open(config.data_file, 'r') as file:
        equips = json.load(file)

    numerical_features = []
    categorical_features = []
    labels = []

    # LabelEncoder 初始化 (在函数内部创建，避免全局变量)
    main_attr_encoder = LabelEncoder()
    aux_attr_encoder = LabelEncoder()
    combined_attr_encoder = LabelEncoder()

    main_attr_encoder.fit(config.main_attributes_options)
    aux_attr_encoder.fit(config.auxiliary_attributes_options + [''])


    # 预先计算所有可能的组合
    combined_attr_options = []
    for main_attr in config.main_attributes_options:
        for aux1 in config.auxiliary_attributes_options + ['']:
            for aux2 in config.auxiliary_attributes_options +['']:
                for aux3 in config.auxiliary_attributes_options + ['']:
                    if aux1 == '' and (aux2 != '' or aux3 != ''):
                        continue
                    if aux2 == '' and aux3 != '':
                        continue
                    combined_attr_options.append(f"{main_attr}_{aux1}_{aux2}_{aux3}")

    combined_attr_encoder.fit(combined_attr_options)

    for equip in equips:
        main_attrs_encoded = main_attr_encoder.transform([equip['main_attrs']])[0]

        combined_attr = f"{equip['main_attrs']}_{equip['agg_added_first']}_{equip['agg_added_second']}_{equip['agg_added_third']}"
        combined_attr_encoded = combined_attr_encoder.transform([combined_attr])[0]

        pay_date = datetime.strptime(equip['pay_date'], "%Y-%m-%dT%H:%M:%SZ")
        year = pay_date.year
        month = pay_date.month
        day = pay_date.day

        numerical_part = [
            equip['repair_num'], year, month, day,
            equip['added_attr_num'], equip['equip_level'],
            equip['serverid'], equip['sale_days'], equip['kindid'],
            equip['main_attrs_value'],
            equip['agg_added_first_value'] if equip['agg_added_first_value'] else 0,
            equip['agg_added_second_value'] if equip['agg_added_second_value'] else 0,
            equip['agg_added_third_value'] if equip['agg_added_third_value'] else 0,
            equip['equip_level'] ** 2,
            equip['main_attrs_value'] ** 2,
            equip['equip_level'] * equip['main_attrs_value'],
            equip['equip_level'] * equip['added_attr_num'],
            equip['main_attrs_value'] * equip['added_attr_num'],
        ]
        numerical_features.append(numerical_part)

        categorical_part = [main_attrs_encoded, combined_attr_encoded]
        categorical_features.append(categorical_part)

        labels.append(equip['price'])

    numerical_features = np.array(numerical_features)
    categorical_features = np.array(categorical_features)
    labels = np.array(labels)

    labels = np.log1p(labels)

    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)

    encoder = OneHotEncoder(sparse_output=False)
    categorical_features_encoded = encoder.fit_transform(categorical_features)

    features = np.hstack([numerical_features_scaled, categorical_features_encoded])

    return features, labels, scaler, encoder, main_attr_encoder, aux_attr_encoder, combined_attr_encoder



def train_model(config, X_train, y_train, X_val, y_val, input_dim):
    """训练模型"""
    model = PricePredictionModel(input_dim, config.hidden_layers, config.dropout_rate).to(config.device)  # 将模型移动到指定设备
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.1, verbose=False)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(config.device),
                                  torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(config.device))  # 数据也移动到设备
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(config.device),
                                torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(config.device))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                y_val_pred = model(batch_X_val)
                loss = criterion(y_val_pred, batch_y_val)
                val_loss += loss.item() * batch_X_val.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
            print(
                f"Epoch [{epoch + 1}/{config.epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}")

    return model


def get_user_input(config, scaler, encoder, main_attr_encoder, aux_attr_encoder, combined_attr_encoder):
    """获取用户输入并进行预处理"""
    print("\n请输入装备的各项属性（根据提示逐个输入）：")

    user_input_data = {}
    user_input_data['repair_num'] = int(input("repair_num（整数类型）: "))
    pay_date_str = input("pay_date（日期格式: YYYY-MM-DDTHH:MM:SSZ）: ")
    pay_date = datetime.strptime(pay_date_str, "%Y-%m-%dT%H:%M:%SZ")
    user_input_data['year'] = pay_date.year
    user_input_data['month'] = pay_date.month
    user_input_data['day'] = pay_date.day
    user_input_data['added_attr_num'] = int(input("added_attr_num（整数类型）: "))
    user_input_data['equip_level'] = int(input("equip_level（整数类型）: "))
    user_input_data['serverid'] = int(input("serverid（整数类型）: "))
    user_input_data['sale_days'] = int(input("sale_days（整数类型）: "))
    user_input_data['kindid'] = int(input("kindid（整数类型）: "))
    user_input_data['main_attrs_value'] = int(input("main_attrs_value（整数类型）: "))
    main_attrs_input = input(f"main_attrs（选择从：{config.main_attributes_options} 中选取）: ")
    user_input_data['main_attrs'] = main_attr_encoder.transform([main_attrs_input])[0]
    user_input_data['agg_added_first_value'] = int(input("agg_added_first_value（整数类型）: "))
    agg_added_first_input = input(f"agg_added_first（选择从：{config.auxiliary_attributes_options} 中选取）: ")
    user_input_data['agg_added_first'] = agg_added_first_input
    user_input_data['agg_added_second_value'] = int(input("agg_added_second_value（整数类型）: "))
    agg_added_second_input = input(f"agg_added_second（选择从：{config.auxiliary_attributes_options} 中选取）: ")
    user_input_data['agg_added_second'] = agg_added_second_input
    user_input_data['agg_added_third_value'] = int(input("agg_added_third_value（整数类型）: "))
    agg_added_third_input = input(f"agg_added_third（选择从：{config.auxiliary_attributes_options} 中选取）: ")
    user_input_data['agg_added_third'] = agg_added_third_input

    combined_attr = f"{main_attrs_input}_{agg_added_first_input}_{agg_added_second_input}_{agg_added_third_input}"
    user_input_data['combined_attr'] = combined_attr_encoder.transform([combined_attr])[0]

    numerical_features = [
        user_input_data['repair_num'],
        user_input_data['year'],
        user_input_data['month'],
        user_input_data['day'],
        user_input_data['added_attr_num'],
        user_input_data['equip_level'],
        user_input_data['serverid'],
        user_input_data['sale_days'],
        user_input_data['kindid'],
        user_input_data['main_attrs_value'],
        user_input_data['agg_added_first_value'],
        user_input_data['agg_added_second_value'],
        user_input_data['agg_added_third_value'],
        user_input_data['equip_level'] ** 2,
        user_input_data['main_attrs_value'] ** 2,
        user_input_data['equip_level'] * user_input_data['main_attrs_value'],
        user_input_data['equip_level'] * user_input_data['added_attr_num'],
        user_input_data['main_attrs_value'] * user_input_data['added_attr_num'],
    ]

    categorical_features = [
        user_input_data['main_attrs'],
        user_input_data['combined_attr']
    ]

    numerical_features_np = np.array(numerical_features).reshape(1, -1)
    numerical_features_scaled = scaler.transform(numerical_features_np)
    categorical_features_np = np.array(categorical_features).reshape(1, -1)
    categorical_features_encoded = encoder.transform(categorical_features_np)
    input_features_scaled = np.hstack([numerical_features_scaled, categorical_features_encoded])
    return torch.tensor(input_features_scaled, dtype=torch.float32).to(config.device) # 将输入也移动到设备


def predict_price(config, model, scaler, encoder, main_attr_encoder, aux_attr_encoder, combined_attr_encoder):
    """进行价格预测"""
    model.eval()
    user_input_tensor = get_user_input(config, scaler, encoder, main_attr_encoder, aux_attr_encoder, combined_attr_encoder)
    with torch.no_grad():
        predicted_price = model(user_input_tensor)
        predicted_price = torch.expm1(predicted_price)
    return predicted_price.item()

def load_model_and_scaler(config, input_dim):
    #加载模型
    model = PricePredictionModel(input_dim, config.hidden_layers, config.dropout_rate)
    model.load_state_dict(torch.load(config.model_path, map_location=config.device)) #, map_location=config.device
    model.to(config.device)
    model.eval() #必须调用eval

    # 加载scaler
    scaler = torch.load(config.scaler_path)
    #加载encoder
    encoder = torch.load(config.encoder_path)
    return model, scaler, encoder

def save_model_and_scaler(config, model, scaler, encoder):
    torch.save(model.state_dict(), config.model_path)
    torch.save(scaler, config.scaler_path)
    torch.save(encoder, config.encoder_path)
    print(f"模型已保存至 {config.model_path}")
    print(f"Scaler已保存至 {config.scaler_path}")
    print(f"Encoder已保存至 {config.encoder_path}")


if __name__ == '__main__':
    # 创建配置实例
    config = Config()

    # 加载和预处理数据
    features, labels, scaler, encoder, main_attr_encoder, aux_attr_encoder, combined_attr_encoder = load_and_preprocess_data(config)

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    input_dim = X_train.shape[1]


    # 训练模式
    if not os.path.exists(config.model_path) or not os.path.exists(config.scaler_path) or not os.path.exists(config.encoder_path):
        # 训练模型
        print("正在训练模型...")
        model = train_model(config, X_train, y_train, X_val, y_val, input_dim)
        # 保存模型和 Scaler
        save_model_and_scaler(config, model, scaler, encoder)


    # 加载模型和 Scaler
    model, scaler, encoder = load_model_and_scaler(config, input_dim)
    print(f"已加载模型: {config.model_path}")
    print(f"已加载 Scaler: {config.scaler_path}")
    print(f"已加载 Encoder: {config.encoder_path}")

    # 进行预测
    predicted_price = predict_price(config, model, scaler, encoder,  main_attr_encoder, aux_attr_encoder, combined_attr_encoder)
    print(f"\n预测的装备价格为：{predicted_price:.2f}元")