import csv

# 定义一个列表，包含5个纽约绿牌（或黄牌或flv）出租车的常用字段（例如：VendorID、lpep_pickup_datetime、trip_distance、fare_amount、total_amount）。

common_fields = [
    "VendorID",
    "lpep_pickup_datetime",
    "trip_distance",
    "fare_amount",
    "total_amount",
]

# 定义一个字典，包含每个字段的含义（例如：VendorID：供应商ID，lpep_pickup_datetime：接客时间，trip_distance：行程距离，fare_amount：车费金额，total_amount：总费用）。


field_descriptions = {
    "VendorID": "供应商ID",
    "lpep_pickup_datetime": "接客时间",
    "trip_distance": "行程距离",
    "fare_amount": "车费金额",
    "total_amount": "总费用",
}

# 使用for循环遍历字典，打印每个字段的名称和含义。


for field, description in field_descriptions.items():
    print(f"{field}: {description}")


# 使用条件语句找出总费用最高的字段，并打印结果
def read_taxi_data(file_path):
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)  # 使用 DictReader 将每一行解析为字典
        taxi_data = []
        for row in reader:
            # 将数值字段转换为浮点数
            row["trip_distance"] = float(row["trip_distance"])
            row["fare_amount"] = float(row["fare_amount"])
            row["total_amount"] = float(row["total_amount"])
            taxi_data.append(row)
        return taxi_data


taxi_data = read_taxi_data("./green_tripdata.csv")
highest_total_amount = max(taxi_data, key=lambda x: x["total_amount"])
print("总费用最高的记录: ")
print(f"总费用: {highest_total_amount['total_amount']}, 记录: {highest_total_amount}")
