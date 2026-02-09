import pandas as pd

data = pd.read_csv(
    "./green_tripdata.csv",
    skiprows=2,
    header=None,
    sep=",",
    skip_blank_lines=True,
    engine="python",
)
columns_to_check = data.columns[:]
data = data.dropna(axis=1, subset=columns_to_check, how="all")
new_column_names = [
    "VendorID",
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "store_and_fwd_flag",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "payment_type",
    "trip_type",
]
data.columns = new_column_names


# 定义一个函数clean_data，用于处理缺失值和异常值。
# 调用这两个函数，对数据进行预处理，并打印前10行数据。


def clean_data(data):
    data = data.dropna(how="all")
    data = data[
        (data["trip_distance"] > 0)
        & (data["fare_amount"] > 0)
        & (data["total_amount"] > 0)
        & (data["passenger_count"] > 0)
    ]
    return data


data = clean_data(data)
print(data.head(10))
# 保存清洗后的数据集
data.to_csv("cleaned_data.csv", index=False)
