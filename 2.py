import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 读取数据文件
file_path = "cleaned_data.csv"
data = pd.read_csv(file_path, parse_dates=["lpep_pickup_datetime"])

# 将时间列按10分钟间隔进行向下取整，创建时间间隔列
data["time_interval"] = data["lpep_pickup_datetime"].dt.floor("10min")
# 按时间间隔分组并统计每个间隔的交通量
flow_data = data.groupby("time_interval").size().reset_index(name="traffic_count")

# 创建完整的时间索引，从最小时间到最大时间，每10分钟一个间隔
full_time_index = pd.date_range(
    start=flow_data["time_interval"].min(),
    end=flow_data["time_interval"].max(),
    freq="10min",
)
# 创建完整的时间序列数据框
full_flow_data = pd.DataFrame({"time_interval": full_time_index})
# 将原始流量数据与完整时间序列合并，填充缺失的时间点
full_flow_data = full_flow_data.merge(flow_data, on="time_interval", how="left")
# 将缺失的交通量填充为0
full_flow_data["traffic_count"] = full_flow_data["traffic_count"].fillna(0)

# 按时间顺序分割训练集和测试集（不打乱顺序）
train_data, test_data = train_test_split(full_flow_data, test_size=0.2, shuffle=False)

# 绘制交通流量随时间变化的图表
plt.figure(figsize=(12, 6))
plt.plot(
    full_flow_data["time_interval"],
    full_flow_data["traffic_count"],
    label="Traffic Count",
    color="blue",
)
plt.xlabel("Time Interval")
plt.ylabel("Traffic Count")
plt.title("Traffic Flow over Time (10-min intervals)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("traffic_flow_plot.png")
plt.show()

# 保存训练集和测试集数据
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

# 线性回归模型部分
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 读取训练集和测试集数据
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# 转换时间间隔列为日期时间格式
train_data["time_interval"] = pd.to_datetime(train_data["time_interval"])
test_data["time_interval"] = pd.to_datetime(test_data["time_interval"])

# 创建时间索引特征（连续数值）
train_data["time_index"] = np.arange(len(train_data))
test_data["time_index"] = np.arange(len(train_data), len(train_data) + len(test_data))

# 添加时间特征
train_data["hour"] = train_data["time_interval"].dt.hour      # 小时
train_data["weekday"] = train_data["time_interval"].dt.weekday # 星期几（0-6，0代表周一）
train_data["month"] = train_data["time_interval"].dt.month    # 月份
train_data["is_weekend"] = (train_data["weekday"] >= 5).astype(int)  # 是否为周末

test_data["hour"] = test_data["time_interval"].dt.hour
test_data["weekday"] = test_data["time_interval"].dt.weekday
test_data["month"] = test_data["time_interval"].dt.month
test_data["is_weekend"] = (test_data["weekday"] >= 5).astype(int)

# 准备特征和标签数据（修改：使用时间索引+时间特征）
X_train = train_data[["time_index", "hour", "weekday", "month", "is_weekend"]].values  # 训练集特征
y_train = train_data["traffic_count"].values  # 训练集标签（交通量）
X_test = test_data[["time_index", "hour", "weekday", "month", "is_weekend"]].values    # 测试集特征
y_test = test_data["traffic_count"].values    # 测试集标签（交通量）

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred = model.predict(X_test)

# 计算模型性能指标
mse = mean_squared_error(y_test, y_pred)  # 均方误差
r2 = r2_score(y_test, y_pred)             # 决定系数

print(f"改进线性回归 - 均方误差 (MSE): {mse:.2f}")
print(f"改进线性回归 - 决定系数 (R²): {r2:.2f}")

# 打印特征重要性
feature_names = ["time_index", "hour", "weekday", "month", "is_weekend"]
coefficients = model.coef_
print("\n线性回归特征系数:")
for name, coef in zip(feature_names, coefficients):
    print(f"  {name}: {coef:.4f}")

# 绘制实际值与预测值的对比图
plt.figure(figsize=(12, 6))
plt.plot(test_data["time_index"], y_test, label="Actual", color="blue", linewidth=1)
plt.plot(
    test_data["time_index"], y_pred, label="Predicted", color="red", linestyle="--", linewidth=1
)
plt.xlabel("Time Index")
plt.ylabel("Traffic Count")
plt.title("Improved Linear Regression: Actual vs Predicted Traffic Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("improved_traffic_prediction_comparison.png")
plt.show()

# 决策树回归模型部分
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取训练集和测试集数据
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# 转换时间间隔列为日期时间格式
train_data["time_interval"] = pd.to_datetime(train_data["time_interval"])
test_data["time_interval"] = pd.to_datetime(test_data["time_interval"])

# 从时间中提取时间特征
train_data["hour"] = train_data["time_interval"].dt.hour      # 小时
train_data["weekday"] = train_data["time_interval"].dt.weekday # 星期几（0-6，0代表周一）
train_data["month"] = train_data["time_interval"].dt.month    # 月份

test_data["hour"] = test_data["time_interval"].dt.hour
test_data["weekday"] = test_data["time_interval"].dt.weekday
test_data["month"] = test_data["time_interval"].dt.month

# 准备特征和标签数据
X_train = train_data[["hour", "weekday", "month"]].values  # 训练集特征（小时、星期、月份）
y_train = train_data["traffic_count"].values               # 训练集标签（交通量）
X_test = test_data[["hour", "weekday", "month"]].values    # 测试集特征（小时、星期、月份）
y_test = test_data["traffic_count"].values                 # 测试集标签（交通量）

# 创建并训练决策树回归模型
model = DecisionTreeRegressor(random_state=114514)  # 设置随机种子保证结果可重现
model.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred = model.predict(X_test)

# 计算模型性能指标
mse = mean_squared_error(y_test, y_pred)  # 均方误差
r2 = r2_score(y_test, y_pred)             # 决定系数

print(f"决策树回归 - 均方误差 (MSE): {mse:.2f}")
print(f"决策树回归 - 决定系数 (R²): {r2:.2f}")

# 绘制实际值与预测值的对比图
plt.figure(figsize=(12, 6))
plt.plot(test_data["time_interval"], y_test, label="Actual", color="blue", linewidth=1)
plt.plot(
    test_data["time_interval"], y_pred, label="Predicted", color="red", linestyle="--", linewidth=1
)
plt.xlabel("Time Interval")
plt.ylabel("Traffic Count")
plt.title("Decision Tree Regression: Actual vs Predicted Traffic Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("decision_tree_prediction_comparison.png")
plt.show()