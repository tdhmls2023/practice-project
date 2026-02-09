import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# 修正：使用tensorflow.keras替代原生keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# ==================== 数据读取与预处理 ====================
# 读取数据集（建议确认文件路径正确）
try:
    df = pd.read_csv("green_tripdata.csv")
except FileNotFoundError:
    print("错误：未找到green_tripdata.csv文件，请检查文件路径！")
    # 生成示例数据（用于测试）
    print("正在生成示例测试数据...")
    np.random.seed(42)
    n_samples = 10000
    df = pd.DataFrame({
        "lpep_pickup_datetime": pd.date_range("2024-01-01", periods=n_samples, freq="1min"),
        "Passenger_count": np.random.randint(1, 6, n_samples),
        "Trip_distance": np.random.uniform(1, 20, n_samples),
        "Payment_type": np.random.randint(1, 5, n_samples),
        "Fare_amount": 2.5 + np.random.uniform(0, 3, n_samples) * np.random.uniform(1, 20, n_samples),
        "VendorID": np.random.randint(1, 3, n_samples),
        "Lpep_dropoff_datetime": pd.date_range("2024-01-01", periods=n_samples, freq="1min") + pd.Timedelta(minutes=10),
        "Store_and_fwd_flag": np.random.choice(["Y", "N"], n_samples),
        "Tip_amount": np.random.uniform(0, 10, n_samples),
        "Extra": np.random.uniform(0, 2, n_samples),
        "MTA_tax": np.random.uniform(0, 1, n_samples),
        "Tolls_amount": np.random.uniform(0, 5, n_samples),
        "Ehail_fee": np.zeros(n_samples),
        "improvement_surcharge": np.random.uniform(0, 1, n_samples),
        "Total_amount": np.random.uniform(5, 100, n_samples)
    })

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

# 独热编码
df = pd.get_dummies(df, columns=["Payment_type"], drop_first=False)

# 删除不必要的列
drop_columns = [
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
]
# 确保列存在再删除
df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

# ==================== 数据质量检查 ====================
print("=== 数据质量检查 ===")
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

# 删除所有NaN
df.dropna(inplace=True)
print(f"删除NaN后数据集形状: {df.shape}")

# 确保Fare_amount列有效
df['Fare_amount'] = pd.to_numeric(df['Fare_amount'], errors='coerce')
df = df[(df['Fare_amount'] > 0) & (df['Fare_amount'] < 1000)]  # 过滤异常车费
df.dropna(subset=['Fare_amount'], inplace=True)
print(f"过滤异常值后最终数据集形状: {df.shape}")

# ==================== 数据标准化与划分 ====================
scaler = StandardScaler()
# 只标准化数值型特征
numeric_features = ["Fare_amount", "distance", "hour", "day_of_week", "Passenger_count"]
numeric_features = [col for col in numeric_features if col in df.columns]
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 分离特征和目标
X = df.drop(["Fare_amount"], axis=1)
y = df["Fare_amount"]

# 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"\n=== 数据集划分 ===")
print(f"训练集: {X_train.shape[0]} 样本, 特征数: {X_train.shape[1]}")
print(f"验证集: {X_val.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# ==================== 模型构建与训练 ====================
print(f"\n=== 模型训练 ===")
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

# 编译模型
model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mae'])

# 早停策略
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    min_delta=0.001,
)

# 训练模型
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1,
)

# ==================== 模型评估 ====================
print(f"\n=== 模型评估 ===")

# 模型预测
y_pred = model.predict(X_test, verbose=0)

# 数据格式统一
y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
y_pred_np = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred

print(f"y_test形状: {y_test_np.shape}, y_pred形状: {y_pred_np.shape}")

# 清理无效预测值
valid_mask = np.isfinite(y_test_np) & np.isfinite(y_pred_np)
y_test_clean = y_test_np[valid_mask]
y_pred_clean = y_pred_np[valid_mask]

if len(y_test_clean) == 0:
    print("错误：没有有效的预测样本！")
else:
    # 计算评估指标
    mae = mean_absolute_error(y_test_clean, y_pred_clean)
    mse = np.mean((y_test_clean - y_pred_clean) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_clean, y_pred_clean)

    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print(f"有效样本数: {len(y_test_clean)}/{len(y_test_np)}")

# ==================== 可视化结果 ====================
# 训练和验证损失曲线
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 训练和验证MAE曲线
plt.subplot(2, 2, 2)
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.title("Training and Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)

# 真实值与预测值对比
plt.subplot(2, 2, 3)
sample_size = min(1000, len(y_test_clean))
plt.scatter(y_test_clean[:sample_size], y_pred_clean[:sample_size], alpha=0.5, s=10)
plt.plot([y_test_clean.min(), y_test_clean.max()],
         [y_test_clean.min(), y_test_clean.max()], 'r--', lw=2)
plt.xlabel("True Values (Standardized)")
plt.ylabel("Predicted Values (Standardized)")
plt.title("True vs Predicted Values")
plt.grid(True)

# 残差图
plt.subplot(2, 2, 4)
residuals = y_test_clean[:sample_size] - y_pred_clean[:sample_size]
plt.scatter(y_pred_clean[:sample_size], residuals, alpha=0.5, s=10)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values (Standardized)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n模型训练完成！")