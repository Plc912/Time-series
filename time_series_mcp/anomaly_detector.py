"""
异常检测模块
实现基于特征的异常检测算法
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.ensemble import IsolationForest
from feature_extractor import extract_features_from_file, AVAILABLE_FEATURES
from utils import load_csv_data, validate_columns, validate_window_size


AVAILABLE_METHODS = ['zscore', 'iqr', 'isolation_forest']


def detect_anomalies_zscore(
    df: pd.DataFrame,
    value_columns: List[str],
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    使用Z-score方法检测异常
    
    参数:
        df: 输入DataFrame
        value_columns: 要检测的数值列列表
        threshold: Z-score阈值（默认3.0，即3倍标准差）
    
    返回:
        包含异常点信息的DataFrame，包含原始df的所有列，加上anomaly_score, method, column列
    """
    anomalies = []
    
    for col in value_columns:
        if col not in df.columns:
            continue
        
        # 计算Z-score
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val == 0:
            continue  # 如果标准差为0，跳过该列
        
        z_scores = np.abs((df[col] - mean_val) / std_val)
        anomaly_mask = z_scores > threshold
        
        # 提取异常点（保留所有原始列）
        anomaly_df = df[anomaly_mask].copy()
        anomaly_df['anomaly_score'] = z_scores[anomaly_mask].values
        anomaly_df['method'] = 'zscore'
        anomaly_df['column'] = col
        anomaly_df['value'] = df.loc[anomaly_mask, col].values
        
        if len(anomaly_df) > 0:
            anomalies.append(anomaly_df)
    
    if not anomalies:
        return pd.DataFrame()
    
    result = pd.concat(anomalies, ignore_index=True)
    return result


def detect_anomalies_iqr(
    df: pd.DataFrame,
    value_columns: List[str],
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    使用IQR（四分位距）方法检测异常
    
    参数:
        df: 输入DataFrame
        value_columns: 要检测的数值列列表
        threshold: IQR倍数阈值（默认1.5）
    
    返回:
        包含异常点信息的DataFrame，包含原始df的所有列，加上anomaly_score, method, column列
    """
    anomalies = []
    
    for col in value_columns:
        if col not in df.columns:
            continue
        
        # 计算四分位数
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            continue  # 如果IQR为0，跳过该列
        
        # 定义异常范围
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # 检测异常
        anomaly_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        # 提取异常点（保留所有原始列）
        anomaly_df = df[anomaly_mask].copy()
        
        # 计算异常分数（距离边界的距离，用IQR归一化）
        anomaly_scores = np.maximum(
            (lower_bound - df[col][anomaly_mask]) / IQR,
            (df[col][anomaly_mask] - upper_bound) / IQR
        )
        
        anomaly_df['anomaly_score'] = anomaly_scores.values
        anomaly_df['method'] = 'iqr'
        anomaly_df['column'] = col
        anomaly_df['value'] = df.loc[anomaly_mask, col].values
        
        if len(anomaly_df) > 0:
            anomalies.append(anomaly_df)
    
    if not anomalies:
        return pd.DataFrame()
    
    result = pd.concat(anomalies, ignore_index=True)
    return result


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    value_columns: List[str],
    threshold: float = 0.1
) -> pd.DataFrame:
    """
    使用Isolation Forest方法检测异常
    
    参数:
        df: 输入DataFrame
        value_columns: 要检测的数值列列表
        threshold: 异常比例（contamination，默认0.1即10%）
    
    返回:
        包含异常点信息的DataFrame
    """
    anomalies = []
    
    for col in value_columns:
        if col not in df.columns:
            continue
        
        # 准备数据（需要至少2个样本）
        data = df[[col]].dropna()
        
        if len(data) < 2:
            continue
        
        # 训练Isolation Forest模型
        # contamination参数控制异常的比例
        contamination = min(max(0.01, threshold), 0.5)  # 限制在1%-50%之间
        model = IsolationForest(contamination=contamination, random_state=42)
        
        # 预测异常
        predictions = model.fit_predict(data)
        anomaly_mask = predictions == -1
        
        # 提取异常点
        anomaly_indices = data.index[anomaly_mask]
        anomaly_df = df.loc[anomaly_indices].copy()
        
        # 计算异常分数（距离分数，转换为正值）
        scores = model.score_samples(data)
        anomaly_scores = -scores[anomaly_mask]  # 负号，使得异常点分数为正
        
        anomaly_df['anomaly_score'] = anomaly_scores
        anomaly_df['method'] = 'isolation_forest'
        anomaly_df['column'] = col
        anomaly_df['value'] = df.loc[anomaly_indices, col].values
        
        if len(anomaly_df) > 0:
            anomalies.append(anomaly_df)
    
    if not anomalies:
        return pd.DataFrame()
    
    result = pd.concat(anomalies, ignore_index=True)
    return result


def detect_anomalies(
    df: pd.DataFrame,
    time_column: str,
    value_columns: List[str],
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    基于提取的特征进行异常检测
    
    参数:
        df: 输入DataFrame（包含时间序列数据）
        time_column: 时间列名称
        value_columns: 要分析的数值列
        method: 异常检测方法 ('zscore', 'iqr', 'isolation_forest')
        threshold: 异常阈值
            - zscore: Z-score阈值（默认3.0）
            - iqr: IQR倍数（默认1.5）
            - isolation_forest: 异常比例（默认0.1）
    
    返回:
        包含异常点信息的DataFrame，包含列：time, column, value, anomaly_score, method
    """
    # 验证方法
    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"不支持的方法 '{method}'。"
            f"可选方法: {', '.join(AVAILABLE_METHODS)}"
        )
    
    # 验证列
    value_columns = validate_columns(df, value_columns)
    
    # 确保有time_column
    if time_column not in df.columns:
        raise ValueError(f"时间列 '{time_column}' 不存在。")
    
    # 根据方法调用相应的检测函数
    if method == 'zscore':
        result = detect_anomalies_zscore(df, value_columns, threshold)
    elif method == 'iqr':
        result = detect_anomalies_iqr(df, value_columns, threshold)
    elif method == 'isolation_forest':
        result = detect_anomalies_isolation_forest(df, value_columns, threshold)
    else:
        raise ValueError(f"未知的方法: {method}")
    
    # 确保结果包含必要的列
    if len(result) > 0:
        # 标准化输出列 - 保留时间列、列名、值、异常分数和方法
        output_columns = []
        
        # 添加时间列
        if time_column in result.columns:
            output_columns.append(time_column)
        
        # 添加列名、值、异常分数和方法
        for col in ['column', 'value', 'anomaly_score', 'method']:
            if col in result.columns:
                output_columns.append(col)
        
        # 只保留需要的列
        if output_columns:
            result = result[output_columns]
    
    return result


def detect_anomalies_from_file(
    file_path: str,
    time_column: str,
    value_columns: List[str],
    method: str = 'zscore',
    threshold: float = 3.0,
    missing_strategy: str = 'interpolate'
) -> pd.DataFrame:
    """
    从CSV文件加载数据并检测异常（便捷函数）
    
    参数:
        file_path: CSV文件路径
        time_column: 时间列名称
        value_columns: 要分析的数值列
        method: 异常检测方法
        threshold: 异常阈值
        missing_strategy: 缺失值处理策略
    
    返回:
        包含异常点信息的DataFrame
    """
    # 加载数据
    df = load_csv_data(file_path, time_column, missing_strategy)
    
    # 检测异常
    return detect_anomalies(df, time_column, value_columns, method, threshold)

