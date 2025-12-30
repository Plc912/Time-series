"""
工具函数模块
提供数据读取、时间解析、数据清理等通用功能
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Union
import warnings


def load_csv_data(file_path: str, time_column: str, missing_strategy: str = 'interpolate') -> pd.DataFrame:
    """
    加载CSV文件并解析时间列
    
    参数:
        file_path: CSV文件路径
        time_column: 时间列名称
        missing_strategy: 缺失值处理策略 ('interpolate', 'drop', 'forward_fill', 'backward_fill')
    
    返回:
        处理后的DataFrame
    
    异常:
        FileNotFoundError: 文件不存在
        ValueError: 时间列格式无法解析
    """
    try:
        # 尝试读取CSV文件
        df = pd.read_csv(file_path)
        
        if time_column not in df.columns:
            raise ValueError(f"时间列 '{time_column}' 在CSV文件中不存在。可用列: {', '.join(df.columns)}")
        
        # 尝试多种时间格式解析
        time_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y',
        ]
        
        parsed = False
        for fmt in time_formats:
            try:
                df[time_column] = pd.to_datetime(df[time_column], format=fmt)
                parsed = True
                break
            except (ValueError, TypeError):
                continue
        
        # 如果所有格式都失败，尝试自动解析
        if not parsed:
            try:
                df[time_column] = pd.to_datetime(df[time_column])
            except Exception as e:
                raise ValueError(f"无法解析时间列 '{time_column}'。请确保时间格式正确。错误: {str(e)}")
        
        # 处理缺失值
        df = handle_missing_values(df, missing_strategy)
        
        # 按时间排序
        df = df.sort_values(by=time_column).reset_index(drop=True)
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 '{file_path}' 不存在。请检查文件路径是否正确。")
    except Exception as e:
        raise ValueError(f"读取CSV文件时出错: {str(e)}")


def handle_missing_values(df: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
    """
    处理DataFrame中的缺失值
    
    参数:
        df: 输入DataFrame
        strategy: 处理策略 ('interpolate', 'drop', 'forward_fill', 'backward_fill')
    
    返回:
        处理后的DataFrame
    """
    df = df.copy()
    
    if strategy == 'interpolate':
        # 线性插值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    elif strategy == 'drop':
        # 删除包含缺失值的行
        df = df.dropna()
    elif strategy == 'forward_fill':
        # 前向填充
        df = df.ffill()
    elif strategy == 'backward_fill':
        # 后向填充
        df = df.bfill()
    else:
        warnings.warn(f"未知的缺失值处理策略 '{strategy}'，将使用默认策略 'interpolate'")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    
    return df


def validate_columns(df: pd.DataFrame, value_columns: List[str]) -> List[str]:
    """
    验证并过滤数值列
    
    参数:
        df: 输入DataFrame
        value_columns: 要验证的列名列表
    
    返回:
        有效的数值列列表
    """
    valid_columns = []
    
    for col in value_columns:
        if col not in df.columns:
            warnings.warn(f"列 '{col}' 不存在，已跳过。")
            continue
        
        # 检查是否为数值类型
        if not pd.api.types.is_numeric_dtype(df[col]):
            # 尝试转换为数值类型
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > 0:  # 如果转换后至少有一些有效值
                    valid_columns.append(col)
                    warnings.warn(f"列 '{col}' 包含非数值数据，已自动转换。")
                else:
                    warnings.warn(f"列 '{col}' 无法转换为数值类型，已跳过。")
            except Exception:
                warnings.warn(f"列 '{col}' 无法转换为数值类型，已跳过。")
        else:
            valid_columns.append(col)
    
    if not valid_columns:
        raise ValueError("没有有效的数值列可用于分析。")
    
    return valid_columns


def validate_window_size(data_length: int, window_size: int, step_size: int = 1) -> tuple:
    """
    验证窗口大小和步长
    
    参数:
        data_length: 数据长度
        window_size: 窗口大小
        step_size: 步长
    
    返回:
        (window_size, step_size) 调整后的窗口大小和步长
    
    异常:
        ValueError: 窗口大小无效
    """
    if window_size <= 0:
        raise ValueError("窗口大小必须大于0。")
    
    if step_size <= 0:
        raise ValueError("步长必须大于0。")
    
    if window_size > data_length:
        warnings.warn(
            f"窗口大小 ({window_size}) 超过数据长度 ({data_length})，"
            f"已自动调整为数据长度。"
        )
        window_size = data_length
    
    if window_size < 2:
        warnings.warn("窗口大小太小，可能导致统计特征计算不准确。建议至少为2。")
    
    return window_size, step_size


def safe_divide(numerator: Union[float, np.ndarray], denominator: Union[float, np.ndarray], default: float = 0.0) -> Union[float, np.ndarray]:
    """
    安全除法，避免除零错误
    
    参数:
        numerator: 分子
        denominator: 分母
        default: 除零时的默认值
    
    返回:
        除法结果
    """
    if isinstance(denominator, (int, float)):
        if denominator == 0:
            return default
        return numerator / denominator
    else:
        result = np.zeros_like(numerator, dtype=float)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator[mask]
        result[~mask] = default
        return result

