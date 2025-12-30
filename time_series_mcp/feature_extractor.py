"""
特征提取模块
实现基于滑动窗口的时间序列特征提取
"""
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from typing import List, Optional, Dict, Any
import warnings
from utils import validate_columns, validate_window_size


# 可用的特征列表
AVAILABLE_FEATURES = [
    'mean', 'variance', 'std', 'quantiles', 'min', 'max', 
    'range', 'skewness', 'kurtosis'
]


def extract_window_features(
    df: pd.DataFrame,
    time_column: str,
    value_columns: List[str],
    window_size: int,
    step_size: int = 1,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    提取滑动窗口特征
    
    参数:
        df: 输入DataFrame
        time_column: 时间列名称
        value_columns: 要分析的数值列列表
        window_size: 窗口大小（时间点数量）
        step_size: 滑动步长（默认1）
        features: 要提取的特征列表（默认全部）
           可选特征: 'mean', 'variance', 'std', 'quantiles', 
                    'min', 'max', 'range', 'skewness', 'kurtosis'
    
    返回:
        包含所有特征的DataFrame
    """
    # 验证列
    value_columns = validate_columns(df, value_columns)
    
    # 验证窗口大小
    window_size, step_size = validate_window_size(len(df), window_size, step_size)
    
    # 默认提取所有特征
    if features is None:
        features = AVAILABLE_FEATURES.copy()
    
    # 验证特征列表
    invalid_features = [f for f in features if f not in AVAILABLE_FEATURES]
    if invalid_features:
        warnings.warn(f"无效的特征: {invalid_features}，已忽略。")
        features = [f for f in features if f in AVAILABLE_FEATURES]
    
    if not features:
        raise ValueError("没有有效的特征可以提取。")
    
    results = []
    
    # 对每个数值列提取特征
    for col in value_columns:
        # 获取数值数据（处理NaN）
        data = df[col].values
        
        # 计算滑动窗口
        num_windows = max(1, (len(data) - window_size) // step_size + 1)
        
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx > len(data):
                break
            
            # 提取窗口数据
            window_data = data[start_idx:end_idx]
            window_data = window_data[~np.isnan(window_data)]  # 移除NaN
            
            # 如果窗口数据为空，跳过
            if len(window_data) == 0:
                continue
            
            # 构建特征字典
            feature_dict = {
                'window_index': i,
                'start_time': df[time_column].iloc[start_idx],
                'end_time': df[time_column].iloc[end_idx - 1],
                'column': col,
                'window_size': len(window_data)
            }
            
            # 提取各类特征
            if 'mean' in features:
                feature_dict['mean'] = float(np.mean(window_data))
            
            if 'variance' in features:
                feature_dict['variance'] = float(np.var(window_data, ddof=1))  # 样本方差
            
            if 'std' in features:
                feature_dict['std'] = float(np.std(window_data, ddof=1))  # 样本标准差
            
            if 'quantiles' in features:
                for q in [25, 50, 75, 95, 99]:
                    feature_dict[f'quantile_{q}'] = float(np.percentile(window_data, q))
            
            if 'min' in features:
                feature_dict['min'] = float(np.min(window_data))
            
            if 'max' in features:
                feature_dict['max'] = float(np.max(window_data))
            
            if 'range' in features:
                feature_dict['range'] = float(np.max(window_data) - np.min(window_data))
            
            if 'skewness' in features:
                if len(window_data) >= 3:  # 偏度至少需要3个数据点
                    feature_dict['skewness'] = float(skew(window_data))
                else:
                    feature_dict['skewness'] = np.nan
            
            if 'kurtosis' in features:
                if len(window_data) >= 4:  # 峰度至少需要4个数据点
                    feature_dict['kurtosis'] = float(kurtosis(window_data))
                else:
                    feature_dict['kurtosis'] = np.nan
            
            results.append(feature_dict)
    
    if not results:
        raise ValueError("未能提取到任何特征。请检查数据、窗口大小和步长设置。")
    
    return pd.DataFrame(results)


def extract_features_from_file(
    file_path: str,
    time_column: str,
    value_columns: List[str],
    window_size: int,
    step_size: int = 1,
    features: Optional[List[str]] = None,
    missing_strategy: str = 'interpolate'
) -> pd.DataFrame:
    """
    从CSV文件加载数据并提取特征（便捷函数）
    
    参数:
        file_path: CSV文件路径
        time_column: 时间列名称
        value_columns: 要分析的数值列列表
        window_size: 窗口大小
        step_size: 滑动步长
        features: 要提取的特征列表
        missing_strategy: 缺失值处理策略
    
    返回:
        包含所有特征的DataFrame
    """
    from utils import load_csv_data
    
    # 加载数据
    df = load_csv_data(file_path, time_column, missing_strategy)
    
    # 提取特征
    return extract_window_features(df, time_column, value_columns, window_size, step_size, features)

