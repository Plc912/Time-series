# 时间序列异常检测 MCP 工具

基于滑动窗口的时间序列特征提取工具，用于异常检测的机器学习分析。

作者：庞力铖
邮箱：3522236586@qq.com

GitHub:https://github.com/Plc912/Time-series.git

## 项目概述

本项目实现了一个基于 FastMCP 框架的时间序列分析工具，提供以下核心功能：

1. **滑动窗口特征提取** - 从时间序列数据中提取统计特征
2. **异常检测** - 使用多种算法检测时间序列中的异常点
3. **特征可视化** - 生成特征的可视化图表

## 技术栈

- **框架**: FastMCP
- **传输协议**: SSE (Server-Sent Events)
- **监听地址**: 127.0.0.1:4008
- **编程语言**: Python 3.10+
- **依赖库**: pandas, numpy, scipy, scikit-learn, matplotlib

## 安装

### 1. 克隆或下载项目

```bash
cd time_series_mcp
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务器

```bash
python server.py
```

服务器将在 `http://127.0.0.1:4008` 上启动，使用 SSE 协议提供服务。

## API 文档

### Tool 1: extract_window_features

提取滑动窗口特征。

**参数:**

- `file_path` (str): CSV文件路径
- `time_column` (str): 时间列名称
- `value_columns` (List[str]): 要分析的数值列列表，例如: `["value", "temperature"]`
- `window_size` (int): 窗口大小（时间点数量）
- `step_size` (int, 可选): 滑动步长，默认为1
- `features` (List[str], 可选): 要提取的特征列表，默认全部
  - 可选特征: `'mean'`, `'variance'`, `'std'`, `'quantiles'`, `'min'`, `'max'`, `'range'`, `'skewness'`, `'kurtosis'`
- `missing_strategy` (str, 可选): 缺失值处理策略，默认 `'interpolate'`
  - `'interpolate'`: 线性插值
  - `'drop'`: 删除缺失值
  - `'forward_fill'`: 前向填充
  - `'backward_fill'`: 后向填充

**返回:**
JSON字符串，包含提取的特征数据。每个记录包含：

- `window_index`: 窗口索引
- `start_time`: 窗口开始时间
- `end_time`: 窗口结束时间
- `column`: 列名
- `window_size`: 窗口内有效数据点数量
- 以及所有提取的特征值（mean, variance, std, quantiles, min, max, range, skewness, kurtosis）

**示例:**

```python
{
  "file_path": "data.csv",
  "time_column": "timestamp",
  "value_columns": ["value"],
  "window_size": 10,
  "step_size": 1
}
```

### Tool 2: detect_anomalies

基于时间序列数据进行异常检测。

**参数:**

- `file_path` (str): CSV文件路径
- `time_column` (str): 时间列名称
- `value_columns` (List[str]): 要分析的数值列列表
- `method` (str, 可选): 异常检测方法，默认 `'zscore'`
  - `'zscore'`: Z-score方法，基于标准差的异常检测
  - `'iqr'`: IQR方法，基于四分位距的异常检测
  - `'isolation_forest'`: Isolation Forest方法，基于机器学习的异常检测
- `threshold` (float, 可选): 异常阈值，默认3.0
  - `zscore`方法: Z-score阈值（默认3.0，即3倍标准差）
  - `iqr`方法: IQR倍数（默认1.5）
  - `isolation_forest`方法: 异常比例（0-1之间，默认0.1即10%）
- `missing_strategy` (str, 可选): 缺失值处理策略，默认 `'interpolate'`

**返回:**
JSON字符串，包含检测到的异常点信息。每个记录包含：

- 时间列（由time_column参数指定）
- `column`: 列名
- `value`: 异常值
- `anomaly_score`: 异常分数
- `method`: 使用的检测方法

**示例:**

```python
{
  "file_path": "data.csv",
  "time_column": "timestamp",
  "value_columns": ["value"],
  "method": "zscore",
  "threshold": 3.0
}
```

### Tool 3: visualize_features

生成特征可视化图表。

**参数:**

- `features_data` (str): 特征数据（JSON格式字符串），通常来自 `extract_window_features` 的输出
- `output_path` (str, 可选): 图表保存路径（目录），默认为 `"./output"`

**返回:**
JSON字符串，包含生成的图表文件路径列表：

- `success`: 是否成功
- `output_path`: 输出目录
- `saved_files`: 保存的文件路径列表
- `count`: 生成的图表数量

**示例:**

```python
{
  "features_data": "...",  # extract_window_features的输出
  "output_path": "./charts"
}
```

## 使用示例

### 1. 生成模拟数据

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 生成时间序列
start_date = datetime(2025, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]
values = np.random.randn(365) * 10 + 50  # 均值为50，标准差为10的正态分布

# 创建DataFrame
df = pd.DataFrame({
    'timestamp': dates,
    'value': values
})

# 保存为CSV
df.to_csv('sample_data.csv', index=False)
print("示例数据已保存到 sample_data.csv")
```

### 2. 提取特征

通过MCP客户端调用 `extract_window_features` 工具：

```json
{
  "file_path": "sample_data.csv",
  "time_column": "timestamp",
  "value_columns": ["value"],
  "window_size": 30,
  "step_size": 1,
  "features": ["mean", "std", "min", "max", "skewness", "kurtosis"]
}
```

### 3. 检测异常

通过MCP客户端调用 `detect_anomalies` 工具：

```json
{
  "file_path": "sample_data.csv",
  "time_column": "timestamp",
  "value_columns": ["value"],
  "method": "zscore",
  "threshold": 3.0
}
```

### 4. 可视化特征

通过MCP客户端调用 `visualize_features` 工具：

```json
{
  "features_data": "...",  # 步骤2的输出
  "output_path": "./output"
}
```

## 特征说明

### 统计特征

- **均值 (mean)**: 窗口内数据的平均值
- **方差 (variance)**: 窗口内数据的方差（样本方差）
- **标准差 (std)**: 窗口内数据的标准差（样本标准差）
- **分位数 (quantiles)**: 25%, 50%, 75%, 95%, 99% 分位数
- **最小值 (min)**: 窗口内数据的最小值
- **最大值 (max)**: 窗口内数据的最大值
- **极差 (range)**: 最大值与最小值的差
- **偏度 (skewness)**: 数据分布的偏度（需要至少3个数据点）
- **峰度 (kurtosis)**: 数据分布的峰度（需要至少4个数据点）

## 异常检测方法

### 1. Z-score 方法

基于标准差的异常检测。计算每个数据点的Z-score（标准分数），如果Z-score的绝对值超过阈值，则判定为异常。

**适用场景**: 数据近似正态分布

**阈值建议**: 3.0（3倍标准差，约0.3%的数据会被标记为异常）

### 2. IQR 方法

基于四分位距（Interquartile Range）的异常检测。使用四分位数计算异常范围，超出范围的数据点被标记为异常。

**适用场景**: 数据分布未知或非正态分布

**阈值建议**: 1.5（标准箱线图方法）

### 3. Isolation Forest 方法

基于机器学习的异常检测。使用Isolation Forest算法，通过隔离异常点来检测异常。

**适用场景**: 复杂的数据分布，多变量异常检测

**阈值建议**: 0.1（10%的数据会被标记为异常，可根据实际情况调整）

## 错误处理

工具提供了完善的错误处理机制：

- **文件不存在**: 返回友好的错误信息，包含文件路径
- **时间格式错误**: 自动尝试多种常见时间格式解析
- **窗口大小无效**: 自动调整或给出警告
- **数值列错误**: 自动过滤非数值数据，给出警告
- **参数错误**: 返回详细的错误信息和建议

## 性能优化

- 使用 numpy 向量化操作提高计算效率
- 大文件支持分块处理（通过pandas的chunk读取）
- 提供进度反馈（通过SSE推送）

## 时间格式支持

工具支持以下常见时间格式的自动识别：

- `%Y-%m-%d %H:%M:%S` (例如: 2025-01-01 12:00:00)
- `%Y-%m-%d` (例如: 2025-01-01)
- `%Y/%m/%d %H:%M:%S` (例如: 2025/01/01 12:00:00)
- `%Y/%m/%d` (例如: 2025/01/01)
- `%d/%m/%Y %H:%M:%S` (例如: 01/01/2025 12:00:00)
- `%d/%m/%Y` (例如: 01/01/2025)
- `%m/%d/%Y %H:%M:%S` (例如: 01/01/2025 12:00:00)
- `%m/%d/%Y` (例如: 01/01/2025)

如果以上格式都无法匹配，工具会尝试pandas的自动解析功能。

## 注意事项

1. **窗口大小**: 窗口大小应小于数据长度。如果窗口大小过大，工具会自动调整并给出警告。
2. **数据质量**: 建议在分析前检查数据质量，处理明显的错误数据。
3. **缺失值**: 默认使用线性插值处理缺失值。对于不同类型的时序数据，可能需要选择不同的策略。
4. **异常检测阈值**: 不同的阈值会产生不同的结果。建议根据业务需求和数据分布特点选择合适的阈值。
5. **可视化**: 可视化功能需要matplotlib后端支持。在某些环境中可能需要配置matplotlib后端。
