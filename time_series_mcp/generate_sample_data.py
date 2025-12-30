"""
生成示例时间序列数据用于测试
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(output_file='sample_data.csv', n_days=365, add_anomalies=True):
    """
    生成示例时间序列数据
    
    参数:
        output_file: 输出文件路径
        n_days: 生成的天数
        add_anomalies: 是否添加异常点
    """
    # 生成时间序列
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # 生成基础数据（正态分布，均值为50，标准差为10）
    np.random.seed(42)  # 设置随机种子以便复现
    values = np.random.randn(n_days) * 10 + 50
    
    # 添加趋势
    trend = np.linspace(0, 20, n_days)
    values = values + trend
    
    # 添加季节性（年度周期）
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    values = values + seasonal
    
    # 可选：添加异常点
    if add_anomalies:
        anomaly_indices = np.random.choice(n_days, size=int(n_days * 0.05), replace=False)
        anomaly_values = np.random.choice([-1, 1], size=len(anomaly_indices)) * np.random.uniform(30, 50, len(anomaly_indices))
        values[anomaly_indices] = values[anomaly_indices] + anomaly_values
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'value': values,
        'temperature': values * 0.5 + 20,  # 添加第二个数值列（温度）
    })
    
    # 保存为CSV
    df.to_csv(output_file, index=False)
    print(f"示例数据已保存到 {output_file}")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"数值列: value, temperature")
    
    return df

if __name__ == "__main__":
    generate_sample_data('sample_data.csv', n_days=365, add_anomalies=True)

