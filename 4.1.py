import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np  # 添加 numpy 导入

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping

# 读取数据集
df = pd.read_csv("green_tripdata.csv")

# 清洗数据：删除缺失值
df.dropna(
    subset=[
        "lpep_pickup_datetime",
        "Passenger_count",
        "Trip_distance",
        "Payment_type",
        "Fare_amount",
    ],
    inplace=True,
)

# 特征提取
df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
df["hour"] = df["lpep_pickup_datetime"].dt.hour
df["day_of_week"] = df["lpep_pickup_datetime"].dt.weekday

# 计算距离
df["distance"] = df["Trip_distance"]

df = pd.get_dummies(df, columns=["Payment_type"])

# 删除不必要的列
df.drop(
    columns=[
        "VendorID",
        "lpep_pickup_datetime",
        "Lpep_dropoff_datetime",
        "Store_and_fwd_flag",
        "Trip_distance",
        "Tip_amount",
        "Extra",
        "MTA_tax",
        "Tolls_amount",
        "Ehail_fee",
        "improvement_surcharge",
        "Total_amount",
    ],
    inplace=True,
)

# ==================== 添加：数据质量检查 ====================
print("数据质量检查:")
print(f"原始数据集形状: {df.shape}")
print(f"数据集中NaN数量: {df.isna().sum().sum()}")
print(f"数据集中Inf数量: {df.isin([np.inf, -np.inf]).sum().sum()}")

# 检查并清理无穷大值
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    inf_count = df[col].isin([np.inf, -np.inf]).sum()
    if inf_count > 0:
        print(f"列 '{col}' 中有 {inf_count} 个无穷大值，已替换为NaN")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# 再次检查
print(f"清理后数据集中NaN数量: {df.isna().sum().sum()}")

# ==================== 修改：删除所有NaN ====================
df.dropna(inplace=True)
print(f"删除NaN后数据集形状: {df.shape}")

# ==================== 修改：分离特征和目标之前检查 ====================
# 确保Fare_amount列没有无效值
print(f"Fare_amount列 - NaN数量: {df['Fare_amount'].isna().sum()}")
print(f"Fare_amount列 - Inf数量: {df['Fare_amount'].isin([np.inf, -np.inf]).sum()}")

# 确保Fare_amount是数值型
df['Fare_amount'] = pd.to_numeric(df['Fare_amount'], errors='coerce')

# 再次检查并删除无效值
df.dropna(subset=['Fare_amount'], inplace=True)
print(f"最终数据集形状: {df.shape}")

scaler = StandardScaler()
df[["Fare_amount", "distance"]] = scaler.fit_transform(df[["Fare_amount", "distance"]])

X = df.drop(["Fare_amount"], axis=1)
y = df["Fare_amount"]

# 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"\n数据集划分:")
print(f"训练集: {X_train.shape[0]} 样本")
print(f"验证集: {X_val.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# ==================== 修改：模型结构优化 ====================
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

# 定义损失函数和优化器
model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mae'])

# 模型训练
early_stopping = EarlyStopping(
    monitor="val_loss",  # 监控验证集损失
    patience=10,  # 增加容忍轮数
    restore_best_weights=True,  # 恢复到最佳权重
    min_delta=0.001,  # 最小变化阈值
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,  # 增加epochs
    batch_size=64,  # 增加batch_size
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1,
)

# ==================== 修改：模型评估与数据清理 ====================
print("\n模型评估:")

# 模型预测
y_pred = model.predict(X_test)

# 确保数据是numpy数组，避免广播问题
y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
y_pred_np = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred

print(f"y_test形状: {y_test_np.shape}, y_pred形状: {y_pred_np.shape}")

# 检查预测结果质量
print(f"y_pred - NaN数量: {np.isnan(y_pred_np).sum()}")
print(f"y_pred - Inf数量: {np.isinf(y_pred_np).sum()}")

# 清理无效预测值
if np.isnan(y_pred_np).any() or np.isinf(y_pred_np).any():
    print("警告: 预测结果包含无效值，正在清理...")

    # 用中位数替换无效值
    median_val = np.nanmedian(y_pred_np)
    y_pred_np = np.where(np.isnan(y_pred_np), median_val, y_pred_np)
    y_pred_np = np.where(np.isinf(y_pred_np), median_val, y_pred_np)

    print(f"清理后 y_pred - NaN数量: {np.isnan(y_pred_np).sum()}")
    print(f"清理后 y_pred - Inf数量: {np.isinf(y_pred_np).sum()}")

# 计算MAE
try:
    mae = mean_absolute_error(y_test_np, y_pred_np)
    print(f"Mean Absolute Error: {mae:.4f}")

    # 计算其他指标
    mse = np.mean((y_test_np - y_pred_np) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_test_np - y_pred_np) ** 2) / np.sum((y_test_np - np.mean(y_test_np)) ** 2))

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

except ValueError as e:
    print(f"计算MAE时出错: {e}")

    # 尝试手动清理
    print("尝试手动清理数据...")
    valid_mask = np.isfinite(y_test_np) & np.isfinite(y_pred_np)
    y_test_clean = y_test_np[valid_mask]
    y_pred_clean = y_pred_np[valid_mask]

    if len(y_test_clean) > 0:
        mae = mean_absolute_error(y_test_clean, y_pred_clean)
        print(f"Mean Absolute Error (基于清理后的数据): {mae:.4f}")
        print(f"使用 {len(y_test_clean)}/{len(y_test_np)} 个有效样本")
    else:
        print("错误: 清理后没有有效的样本！")

# 可视化结果
# 训练和验证损失曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 真实值与预测值对比
plt.subplot(1, 2, 2)
plt.scatter(y_test_np[:1000], y_pred_np[:1000], alpha=0.5, s=10)  # 只显示前1000个点
plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.grid(True)

plt.tight_layout()
plt.show()

# ==================== 添加：残差图 ====================
plt.figure(figsize=(8, 6))
residuals = y_test_np - y_pred_np
plt.scatter(y_pred_np[:1000], residuals[:1000], alpha=0.5, s=10)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

print("\n模型训练完成！")
