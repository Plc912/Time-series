"""
FastMCP服务器主文件
提供时间序列特征提取和异常检测的MCP工具服务
"""
import json
import pandas as pd
from typing import List, Optional, Any
from fastmcp import FastMCP
from feature_extractor import extract_features_from_file
from anomaly_detector import detect_anomalies_from_file
from visualizer import visualize_features as generate_visualizations


# 创建FastMCP实例
mcp = FastMCP("Time Series Feature Extractor", debug=True, log_level="INFO")


@mcp.tool()
def extract_window_features(
    file_path: str,
    time_column: str,
    value_columns: List[str],
    window_size: int,
    step_size: int = 1,
    features: Optional[List[str]] = None,
    missing_strategy: str = 'interpolate'
) -> str:
    """
    提取滑动窗口特征
    
    从CSV文件中读取时间序列数据，使用滑动窗口提取统计特征。
    
    参数:
        file_path: CSV文件路径
        time_column: 时间列名称
        value_columns: 要分析的数值列列表，例如: ["value", "temperature"]
        window_size: 窗口大小（整数，表示时间点数量）
        step_size: 滑动步长（默认为1，表示每次移动1个时间点）
        features: 要提取的特征列表（默认全部）
            可选特征: 'mean', 'variance', 'std', 'quantiles', 
                     'min', 'max', 'range', 'skewness', 'kurtosis'
        missing_strategy: 缺失值处理策略
            - 'interpolate': 线性插值（默认）
            - 'drop': 删除缺失值
            - 'forward_fill': 前向填充
            - 'backward_fill': 后向填充
    
    返回:
        JSON字符串，包含提取的特征数据。每个记录包含窗口的时间范围、
        列名、窗口索引和所有提取的特征值。
    
    示例:
        ```python
        result = extract_window_features(
            file_path="data.csv",
            time_column="timestamp",
            value_columns=["value"],
            window_size=10,
            step_size=1
        )
        ```
    """
    try:
        # 提取特征
        features_df = extract_features_from_file(
            file_path=file_path,
            time_column=time_column,
            value_columns=value_columns,
            window_size=window_size,
            step_size=step_size,
            features=features,
            missing_strategy=missing_strategy
        )
        
        # 转换为JSON格式
        # 处理时间类型，使其可序列化
        result_dict = features_df.to_dict(orient='records')
        for record in result_dict:
            # 将datetime对象转换为字符串
            for key, value in record.items():
                if hasattr(value, 'isoformat'):  # 如果是datetime对象
                    record[key] = value.isoformat()
        
        return json.dumps(result_dict, ensure_ascii=False, indent=2)
    
    except FileNotFoundError as e:
        error_msg = {
            "error": "文件不存在",
            "message": str(e),
            "file_path": file_path
        }
        return json.dumps(error_msg, ensure_ascii=False)
    
    except ValueError as e:
        error_msg = {
            "error": "参数错误",
            "message": str(e)
        }
        return json.dumps(error_msg, ensure_ascii=False)
    
    except Exception as e:
        error_msg = {
            "error": "处理失败",
            "message": str(e),
            "type": type(e).__name__
        }
        return json.dumps(error_msg, ensure_ascii=False)


@mcp.tool()
def detect_anomalies(
    file_path: str,
    time_column: str,
    value_columns: List[str],
    method: str = 'zscore',
    threshold: Optional[float] = None,
    missing_strategy: str = 'interpolate'
) -> str:
    """
    基于时间序列数据进行异常检测
    
    从CSV文件中读取时间序列数据，使用指定的方法检测异常点。
    
    参数:
        file_path: CSV文件路径
        time_column: 时间列名称
        value_columns: 要分析的数值列列表，例如: ["value", "temperature"]
        method: 异常检测方法
            - 'zscore': Z-score方法（默认），基于标准差的异常检测
            - 'iqr': IQR方法，基于四分位距的异常检测
            - 'isolation_forest': Isolation Forest方法，基于机器学习的异常检测
        threshold: 异常阈值（可选，默认根据method自动设置）
            - zscore方法: Z-score阈值，默认3.0（即3倍标准差）
            - iqr方法: IQR倍数，默认1.5
            - isolation_forest方法: 异常比例（0-1之间），默认0.1（即10%）
            - 如果不指定，将根据method自动使用合适的默认值
        missing_strategy: 缺失值处理策略
            - 'interpolate': 线性插值（默认）
            - 'drop': 删除缺失值
            - 'forward_fill': 前向填充
            - 'backward_fill': 后向填充
    
    返回:
        JSON字符串，包含检测到的异常点信息。每个记录包含时间、列名、
        异常值、异常分数和使用的检测方法。
    
    示例:
        ```python
        result = detect_anomalies(
            file_path="data.csv",
            time_column="timestamp",
            value_columns=["value"],
            method="zscore",
            threshold=3.0
        )
        ```
    """
    try:
        # 根据方法自动设置threshold默认值
        if threshold is None:
            if method == 'zscore':
                threshold = 3.0
            elif method == 'iqr':
                threshold = 1.5
            elif method == 'isolation_forest':
                threshold = 0.1
            else:
                threshold = 3.0  # 默认值
        
        # 检测异常
        anomalies_df = detect_anomalies_from_file(
            file_path=file_path,
            time_column=time_column,
            value_columns=value_columns,
            method=method,
            threshold=threshold,
            missing_strategy=missing_strategy
        )
        
        # 转换为JSON格式
        result_dict = anomalies_df.to_dict(orient='records')
        for record in result_dict:
            # 将datetime对象转换为字符串
            for key, value in record.items():
                if hasattr(value, 'isoformat'):  # 如果是datetime对象
                    record[key] = value.isoformat()
                elif pd.isna(value):  # 处理NaN值
                    record[key] = None
        
        return json.dumps(result_dict, ensure_ascii=False, indent=2)
    
    except FileNotFoundError as e:
        error_msg = {
            "error": "文件不存在",
            "message": str(e),
            "file_path": file_path
        }
        return json.dumps(error_msg, ensure_ascii=False)
    
    except ValueError as e:
        error_msg = {
            "error": "参数错误",
            "message": str(e)
        }
        return json.dumps(error_msg, ensure_ascii=False)
    
    except Exception as e:
        error_msg = {
            "error": "处理失败",
            "message": str(e),
            "type": type(e).__name__
        }
        return json.dumps(error_msg, ensure_ascii=False)


@mcp.tool()
def visualize_features(
    features_data: str,
    output_path: str = "./output"
) -> str:
    """
    生成特征可视化图表
    
    根据提取的特征数据生成可视化图表，支持多种统计特征的展示。
    
    参数:
        features_data: 特征数据（JSON格式字符串），通常来自extract_window_features的输出
        output_path: 图表保存路径（目录），默认为"./output"
    
    返回:
        JSON字符串，包含生成的图表文件路径列表
    
    示例:
        ```python
        # 先提取特征
        features_json = extract_window_features(...)
        
        # 然后可视化
        result = visualize_features(
            features_data=features_json,
            output_path="./charts"
        )
        ```
    """
    try:
        # 解析JSON数据
        if isinstance(features_data, str):
            data = json.loads(features_data)
        else:
            data = features_data
        
        # 生成可视化图表
        saved_paths = generate_visualizations(data, output_path)
        
        # 返回结果
        result = {
            "success": True,
            "output_path": output_path,
            "saved_files": saved_paths,
            "count": len(saved_paths)
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except json.JSONDecodeError as e:
        error_msg = {
            "error": "JSON解析错误",
            "message": str(e)
        }
        return json.dumps(error_msg, ensure_ascii=False)
    
    except ValueError as e:
        error_msg = {
            "error": "参数错误",
            "message": str(e)
        }
        return json.dumps(error_msg, ensure_ascii=False)
    
    except Exception as e:
        error_msg = {
            "error": "处理失败",
            "message": str(e),
            "type": type(e).__name__
        }
        return json.dumps(error_msg, ensure_ascii=False)


if __name__ == "__main__":
    # 运行FastMCP服务器
    # 监听地址: 127.0.0.1:4008
    # 使用SSE (Server-Sent Events) 传输协议
    mcp.run(transport="sse", port=4008)

