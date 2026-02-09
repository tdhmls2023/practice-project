# -*- coding: utf-8 -*-
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os

# 设置中文字体
rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

class TaxiDataAnalyzer:
    """出租车数据分析器"""
    
    def __init__(self):
        # 出租车常用字段
        self.common_fields = [
            "VendorID", "lpep_pickup_datetime", "Trip_distance", 
            "Fare_amount", "Total_amount"
        ]
        # 字典
        self.field_descriptions = {
            "VendorID": "供应商ID",
            "lpep_pickup_datetime": "接客时间",
            "Trip_distance": "行程距离",
            "Fare_amount": "车费金额", 
            "Total_amount": "总费用",
        }
    
    def print_field_descriptions(self):
        """打印字段名称和含义"""
        print("=== 出租车数据字段说明 ===")
        for field, description in self.field_descriptions.items():
            print(f"{field}: {description}")
        print()
    
    def read_taxi_data_csv(self, file_path):
        """使用csv模块读取出租车数据"""
        try:
            with open(file_path, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file) # 将每行数据转换为字典格式
                taxi_data = []
                
                for row_num, row in enumerate(reader, 1):
                    try:
                        # 安全地转换数值字段
                        processed_row = {}
                        for key, value in row.items():
                            if key in ["Trip_distance", "Fare_amount", "Total_amount"]:
                                processed_row[key] = float(value) if value.strip() else 0.0
                            else:
                                processed_row[key] = value
                        taxi_data.append(processed_row)
                        
                    except (ValueError, KeyError) as e:
                        print(f"警告: 第{row_num}行数据格式错误: {e}")
                        continue
                
                print(f"成功读取 {len(taxi_data)} 条记录")
                return taxi_data
                
        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 不存在")
            return []
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return []
    
    def find_highest_total_amount(self, taxi_data):
        """找出总费用最高的记录"""
        if not taxi_data:
            print("没有数据可供分析")
            return None
        
        try:
            highest_record = max(taxi_data, key=lambda x: x.get("Total_amount", 0))
            print("\n=== 总费用最高的记录 ===")
            print(f"总费用: {highest_record['Total_amount']:.2f}")
            
            # 只显示相关字段
            relevant_fields = {k: v for k, v in highest_record.items() 
                             if k in self.common_fields}
            for field, value in relevant_fields.items():
                description = self.field_descriptions.get(field, field)
                print(f"{description}: {value}")
            
            return highest_record
            
        except Exception as e:
            print(f"分析最高费用时发生错误: {e}")
            return None
    
    def read_taxi_data_pandas(self, file_path):
        """使用pandas读取出租车数据"""
        try:
            # 尝试不同的读取方式
            try:
                data = pd.read_csv(file_path)
            except pd.errors.ParserError:
                # 如果标准读取失败，尝试跳过前几行
                data = pd.read_csv(file_path, skiprows=2, header=None, engine='python')
                
                # 定义列名
                new_column_names = [
                    "VendorID", "lpep_pickup_datetime", "lpep_dropoff_datetime",
                    "store_and_fwd_flag", "RatecodeID", "PULocationID", "DOLocationID",
                    "passenger_count", "Trip_distance", "Fare_amount", "extra",
                    "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
                    "Total_amount", "payment_type", "trip_type"
                ]
                
                # 只保留与列名数量匹配的列
                data = data.iloc[:, :len(new_column_names)]
                data.columns = new_column_names[:data.shape[1]]
            
            print(f"Pandas读取数据形状: {data.shape}")
            return data
            
        except Exception as e:
            print(f"Pandas读取文件时发生错误: {e}")
            return pd.DataFrame()
    
    def clean_data(self, data):
        """数据清洗函数"""
        if data.empty:
            return data
        
        print("\n=== 数据清洗过程 ===")
        initial_shape = data.shape
        print(f"原始数据形状: {initial_shape}")
        
        # 1. 删除全为空值的行
        data_cleaned = data.dropna(how='all')
        print(f"删除全空行后: {data_cleaned.shape}")
        
        # 2. 处理数值型字段的异常值
        numeric_conditions = [
            data_cleaned["Trip_distance"] > 0,
            data_cleaned["Fare_amount"] > 0,
            data_cleaned["Total_amount"] > 0
        ]
        
        # 如果存在passenger_count列，也进行检查
        if "passenger_count" in data_cleaned.columns:
            numeric_conditions.append(data_cleaned["passenger_count"] > 0)
        
        # 组合所有条件
        valid_mask = pd.concat(numeric_conditions, axis=1).all(axis=1)
        data_cleaned = data_cleaned[valid_mask]
        
        print(f"删除异常值后: {data_cleaned.shape}")
        print(f"总共删除记录: {initial_shape[0] - data_cleaned.shape[0]}")
        
        return data_cleaned
    
    def calculate_statistics(self, data):
        """计算统计量"""
        print("\n=== 数据统计信息 ===")
        
        stats = {}
        if not data.empty:
            # 行程距离统计
            trip_stats = {
                "平均值": data["Trip_distance"].mean(),
                "最大值": data["Trip_distance"].max(),
                "最小值": data["Trip_distance"].min(),
                "标准差": data["Trip_distance"].std()
            }
            
            # 车费金额统计
            fare_stats = {
                "平均值": data["Fare_amount"].mean(),
                "最大值": data["Fare_amount"].max(),
                "最小值": data["Fare_amount"].min(),
                "标准差": data["Fare_amount"].std()
            }
            
            print("行程距离统计:")
            for key, value in trip_stats.items():
                print(f"  {key}: {value:.2f}")
            
            print("\n车费金额统计:")
            for key, value in fare_stats.items():
                print(f"  {key}: {value:.2f}")
            
            stats = {"Trip_distance": trip_stats, "Fare_amount": fare_stats}
        
        return stats
    
    def plot_histograms(self, data):
        """绘制直方图"""
        if data.empty:
            print("没有数据可绘制直方图")
            return
        
        plt.figure(figsize=(14, 6))
        
        # 行程距离直方图
        plt.subplot(1, 2, 1)
        plt.hist(data["Trip_distance"], bins=30, color="skyblue", 
                edgecolor="black", alpha=0.7)
        plt.title("行程距离分布", fontsize=14)
        plt.xlabel("行程距离", fontsize=12)
        plt.ylabel("频率", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 车费金额直方图
        plt.subplot(1, 2, 2)
        plt.hist(data["Fare_amount"], bins=30, color="lightcoral", 
                edgecolor="black", alpha=0.7)
        plt.title("车费金额分布", fontsize=14)
        plt.xlabel("车费金额", fontsize=12)
        plt.ylabel("频率", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_temporal_trends(self, data):
        """分析时间趋势"""
        if data.empty or "lpep_pickup_datetime" not in data.columns:
            print("无法进行时间趋势分析：缺少时间字段")
            return
        
        try:
            # 转换时间字段
            data["lpep_pickup_datetime"] = pd.to_datetime(
                data["lpep_pickup_datetime"], errors='coerce'
            )
            
            # 删除转换失败的行
            data = data.dropna(subset=["lpep_pickup_datetime"])
            
            # 按天聚合数据
            daily_data = data.groupby(data["lpep_pickup_datetime"].dt.date).agg({
                "Trip_distance": "sum",
                "Fare_amount": "sum",
                "Total_amount": "count"  # 行程数量
            }).rename(columns={"Total_amount": "trip_count"})
            
            if len(daily_data) < 2:
                print("数据量不足，无法显示趋势")
                return
            
            # 绘制趋势图
            plt.figure(figsize=(12, 8))
            
            # 行程距离趋势
            plt.subplot(2, 1, 1)
            plt.plot(daily_data.index, daily_data["Trip_distance"], 
                    marker='o', linewidth=2, markersize=4, color='blue')
            plt.title("每日总行程距离趋势", fontsize=14)
            plt.ylabel("总行程距离", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # 车费金额趋势
            plt.subplot(2, 1, 2)
            plt.plot(daily_data.index, daily_data["Fare_amount"], 
                    marker='s', linewidth=2, markersize=4, color='red')
            plt.title("每日总车费金额趋势", fontsize=14)
            plt.ylabel("总车费金额", fontsize=12)
            plt.xlabel("日期", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            return daily_data
            
        except Exception as e:
            print(f"时间趋势分析时发生错误: {e}")
            return None

def main():
    """主函数"""
    analyzer = TaxiDataAnalyzer()
    
    # 子任务1：基础语法和数据结构
    print("=" * 50)
    analyzer.print_field_descriptions()
    
    # 读取数据并分析最高费用
    taxi_data = analyzer.read_taxi_data_csv("./green_tripdata.csv")
    analyzer.find_highest_total_amount(taxi_data)
    
    # 子任务2：数据预处理
    print("\n" + "=" * 50)
    print("子任务2：数据预处理")
    
    data = analyzer.read_taxi_data_pandas("./green_tripdata.csv")
    if not data.empty:
        data_cleaned = analyzer.clean_data(data)
        print("\n清洗后的前10行数据:")
        print(data_cleaned.head(10))
        
        # 保存清洗后的数据
        data_cleaned.to_csv("cleaned_data.csv", index=False)
        print(f"\n清洗后的数据已保存到: cleaned_data.csv")
        
        # 子任务3：统计分析
        print("\n" + "=" * 50)
        print("子任务3：统计分析")
        
        # 计算统计量
        stats = analyzer.calculate_statistics(data_cleaned)
        
        # 绘制直方图
        analyzer.plot_histograms(data_cleaned)
        
        # 时间趋势分析
        print("\n时间趋势分析")
        analyzer.analyze_temporal_trends(data_cleaned)
    
    print("\n" + "=" * 50)
    print("所有任务完成！")

if __name__ == "__main__":
    main()
