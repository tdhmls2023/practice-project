import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

data = pd.read_csv("./cleaned_data.csv")

# 设置 Matplotlib 使用的中文字体
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

# 计算行程距离和车费金额的平均值、最大值和最小值，并打印结果。
trip_distance_stats = {
    "平均值": data["trip_distance"].mean(),
    "最大值": data["trip_distance"].max(),
    "最小值": data["trip_distance"].min(),
}
fare_amount_stats = {
    "平均值": data["fare_amount"].mean(),
    "最大值": data["fare_amount"].max(),
    "最小值": data["fare_amount"].min(),
}

print("行程距离统计值：", trip_distance_stats)
print("车费金额统计值：", fare_amount_stats)

# 使用Matplotlib库绘制行程距离和车费金额的直方图，观察分布情况
# 绘制行程距离的直方图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data["trip_distance"], bins=50, color="blue", edgecolor="black")
plt.title("行程距离的直方图")
plt.xlabel("行程距离")
plt.ylabel("频率")

# 绘制车费金额的直方图
plt.subplot(1, 2, 2)
plt.hist(data["fare_amount"], bins=50, color="green", edgecolor="black")
plt.title("车费金额的直方图")
plt.xlabel("车费金额")
plt.ylabel("频率")

plt.tight_layout()
plt.show()

# 计算每日行程距离和车费金额的变化趋势，并绘制折线图，观察周期性变化。
data["lpep_pickup_datetime"] = pd.to_datetime(data["lpep_pickup_datetime"])

# 按天聚合数据，计算每日的行程距离和车费金额的总和
daily_data = data.groupby(data["lpep_pickup_datetime"].dt.date)[
    ["trip_distance", "fare_amount"]
].sum()

plt.figure(figsize=(10, 6))
plt.plot(
    daily_data.index,
    daily_data["trip_distance"],
    label="行程距离",
    color="blue",
    marker="o",
)
plt.plot(
    daily_data.index,
    daily_data["fare_amount"],
    label="车费金额",
    color="green",
    marker="x",
)
plt.title("每日行程距离和车费金额变化趋势")
plt.xlabel("日期")
plt.ylabel("总量")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
