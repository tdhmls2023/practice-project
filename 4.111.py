import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
import math

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# 1. 数据获取与预处理
def load_and_preprocess_data(file_path):
    # 加载数据
    df = pd.read_csv(file_path)

    # 数据清洗
    # 移除关键字段的缺失值
    df = df.dropna(subset=['lpep_pickup_datetime', 'Lpep_dropoff_datetime',
                           'Pickup_longitude', 'Pickup_latitude',
                           'Dropoff_longitude', 'Dropoff_latitude',
                           'Trip_distance', 'Fare_amount', 'Total_amount'])

    # 移除异常值
    # 行程距离和费用不能为负
    df = df[(df['Trip_distance'] > 0) & (df['Fare_amount'] > 0)]

    # 移除极端异常值（行程时间超过6小时或费用超过500）
    df['pickup_dt'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['dropoff_dt'] = pd.to_datetime(df['Lpep_dropoff_datetime'])
    df['trip_duration'] = (df['dropoff_dt'] - df['pickup_dt']).dt.total_seconds() / 60  # 分钟

    df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 360)]  # 0-6小时
    df = df[df['Fare_amount'] < 500]

    # 特征工程
    # 时间特征
    df['hour'] = df['pickup_dt'].dt.hour
    df['day_of_week'] = df['pickup_dt'].dt.dayofweek
    df['month'] = df['pickup_dt'].dt.month

    # 计算起点和终点的曼哈顿距离（近似距离）
    df['manhattan_distance'] = abs(df['Pickup_longitude'] - df['Dropoff_longitude']) + \
                               abs(df['Pickup_latitude'] - df['Dropoff_latitude'])

    # 计算直线距离（Haversine距离）
    def haversine_distance(lat1, lon1, lat2, lon2):
        # 将十进制度数转化为弧度
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        # haversine公式
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        # 地球半径（千米）
        r = 6371
        return c * r

    df['straight_distance'] = df.apply(lambda row: haversine_distance(
        row['Pickup_latitude'], row['Pickup_longitude'],
        row['Dropoff_latitude'], row['Dropoff_longitude']), axis=1)

    # 速度特征（km/h）
    df['speed'] = (df['straight_distance'] / 1000) / (df['trip_duration'] / 60)
    # 移除明显不合理的速度（如超过120km/h）
    df = df[df['speed'] < 120]

    # 类别型变量处理
    # 支付类型 one-hot编码
    payment_type = pd.get_dummies(df['Payment_type'], prefix='payment_type')
    df = pd.concat([df, payment_type], axis=1)

    # 选择特征和目标变量
    features = ['Pickup_longitude', 'Pickup_latitude',
                'Dropoff_longitude', 'Dropoff_latitude',
                'Trip_distance', 'manhattan_distance', 'straight_distance',
                'hour', 'day_of_week', 'month', 'Passenger_count',
                'Payment_type']

    # 目标变量：行程时间和费用
    targets = ['trip_duration', 'Fare_amount']

    # 标准化连续特征
    continuous_features = ['Pickup_longitude', 'Pickup_latitude',
                           'Dropoff_longitude', 'Dropoff_latitude',
                           'Trip_distance', 'manhattan_distance', 'straight_distance',
                           'hour', 'day_of_week', 'month', 'Passenger_count']

    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    return df[features], df[targets], scaler


# 2. 自定义数据集类
class TaxiDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# 3. 模型定义
class TaxiFarePredictor(nn.Module):
    def __init__(self, input_size):
        super(TaxiFarePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.output_duration = nn.Linear(64, 1)  # 行程时间预测
        self.output_fare = nn.Linear(64, 1)  # 费用预测

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))

        duration = self.output_duration(x)
        fare = self.output_fare(x)

        return duration, fare

# 4. 训练和评估函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=5):
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # 前向传播
            pred_duration, pred_fare = model(inputs)

            # 计算损失（同时预测行程时间和费用）
            loss1 = criterion(pred_duration, targets[:, 0].unsqueeze(1))  # 行程时间
            loss2 = criterion(pred_fare, targets[:, 1].unsqueeze(1))  # 费用
            loss = loss1 + loss2

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                pred_duration, pred_fare = model(inputs)
                loss1 = criterion(pred_duration, targets[:, 0].unsqueeze(1))
                loss2 = criterion(pred_fare, targets[:, 1].unsqueeze(1))
                loss = loss1 + loss2
                val_loss += loss.item()

        # 记录损失
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_loss_history, val_loss_history


def evaluate_model(model, test_loader):
    model.eval()
    all_preds_duration = []
    all_preds_fare = []
    all_targets_duration = []
    all_targets_fare = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            pred_duration, pred_fare = model(inputs)

            all_preds_duration.extend(pred_duration.squeeze().tolist())
            all_preds_fare.extend(pred_fare.squeeze().tolist())
            all_targets_duration.extend(targets[:, 0].tolist())
            all_targets_fare.extend(targets[:, 1].tolist())

    # 计算指标
    mae_duration = mean_absolute_error(all_targets_duration, all_preds_duration)
    rmse_duration = math.sqrt(mean_squared_error(all_targets_duration, all_preds_duration))

    mae_fare = mean_absolute_error(all_targets_fare, all_preds_fare)
    rmse_fare = math.sqrt(mean_squared_error(all_targets_fare, all_preds_fare))

    print(f'Duration Prediction - MAE: {mae_duration:.2f} minutes, RMSE: {rmse_duration:.2f} minutes')
    print(f'Fare Prediction - MAE: ${mae_fare:.2f}, RMSE: ${rmse_fare:.2f}')

    return all_preds_duration, all_preds_fare, all_targets_duration, all_targets_fare


# 5. 主函数
def main():
    # 加载和预处理数据
    print("Loading and preprocessing data...")
    features, targets, scaler = load_and_preprocess_data("E:/Pycharm-Project/AIB/green_tripdata.csv")

    # 仅使用前10%的数据进行快速测试
    n_samples = int(0.1 * len(features))
    features = features.iloc[:n_samples]
    targets = targets.iloc[:n_samples]

    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(features, targets, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 创建数据集和数据加载器
    train_dataset = TaxiDataset(X_train, y_train)
    val_dataset = TaxiDataset(X_val, y_val)
    test_dataset = TaxiDataset(X_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    input_size = X_train.shape[1]
    model = TaxiFarePredictor(input_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("Training model...")
    model, train_loss_history, val_loss_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=100)

    # 绘制学习曲线
    plt.figure(figsize=(12, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 评估模型
    print("Evaluating model...")
    preds_duration, preds_fare, targets_duration, targets_fare = evaluate_model(model, test_loader)

    # 可视化预测结果
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(targets_duration, preds_duration, alpha=0.3)
    plt.plot([min(targets_duration), max(targets_duration)],
             [min(targets_duration), max(targets_duration)], 'r--')
    plt.xlabel('Actual Duration (minutes)')
    plt.ylabel('Predicted Duration (minutes)')
    plt.title('Duration Prediction')

    plt.subplot(1, 2, 2)
    plt.scatter(targets_fare, preds_fare, alpha=0.3)
    plt.plot([min(targets_fare), max(targets_fare)],
             [min(targets_fare), max(targets_fare)], 'r--')
    plt.xlabel('Actual Fare ($)')
    plt.ylabel('Predicted Fare ($)')
    plt.title('Fare Prediction')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()