"""
可视化模块
生成特征可视化图表
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import json
from typing import Dict, Any, List
import os


def visualize_features(features_data: Dict[str, Any], output_path: str) -> List[str]:
    """
    生成特征可视化图表
    
    参数:
        features_data: 特征数据（可以是JSON字符串或字典）
        output_path: 图表保存路径（目录）
    
    返回:
        图表文件路径列表
    """
    # 如果输入是JSON字符串，先解析
    if isinstance(features_data, str):
        features_data = json.loads(features_data)
    
    # 转换为DataFrame
    if isinstance(features_data, list):
        df = pd.DataFrame(features_data)
    elif isinstance(features_data, dict):
        df = pd.DataFrame([features_data])
    else:
        df = pd.DataFrame(features_data)
    
    if len(df) == 0:
        raise ValueError("特征数据为空，无法生成可视化图表。")
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 获取数值特征列（排除标识列）
    exclude_columns = ['window_index', 'start_time', 'end_time', 'column', 'window_size']
    feature_columns = [col for col in df.columns if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not feature_columns:
        raise ValueError("没有找到可可视化的数值特征列。")
    
    saved_paths = []
    
    # 如果有column列，按列分组绘制
    if 'column' in df.columns:
        for column_name in df['column'].unique():
            column_df = df[df['column'] == column_name].copy()
            
            # 使用start_time作为x轴
            if 'start_time' in column_df.columns:
                try:
                    # 尝试转换为datetime
                    column_df['start_time'] = pd.to_datetime(column_df['start_time'])
                    x_data = column_df['start_time']
                    x_label = '时间'
                except:
                    x_data = column_df['start_time']
                    x_label = '开始时间'
            else:
                x_data = column_df.index
                x_label = '窗口索引'
            
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'特征可视化 - {column_name}', fontsize=16, fontweight='bold')
            
            # 绘制统计特征
            ax = axes[0, 0]
            if 'mean' in feature_columns:
                ax.plot(x_data, column_df['mean'], label='均值', linewidth=2)
            if 'std' in feature_columns:
                ax.fill_between(
                    x_data,
                    column_df['mean'] - column_df['std'] if 'mean' in feature_columns and 'std' in feature_columns else x_data,
                    column_df['mean'] + column_df['std'] if 'mean' in feature_columns and 'std' in feature_columns else x_data,
                    alpha=0.3, label='±1标准差'
                )
            ax.set_xlabel(x_label)
            ax.set_ylabel('值')
            ax.set_title('均值和标准差')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 绘制极值
            ax = axes[0, 1]
            if 'min' in feature_columns and 'max' in feature_columns:
                ax.plot(x_data, column_df['min'], label='最小值', linewidth=1.5, alpha=0.7)
                ax.plot(x_data, column_df['max'], label='最大值', linewidth=1.5, alpha=0.7)
                if 'mean' in feature_columns:
                    ax.plot(x_data, column_df['mean'], label='均值', linewidth=2)
                ax.fill_between(x_data, column_df['min'], column_df['max'], alpha=0.2, label='范围')
            ax.set_xlabel(x_label)
            ax.set_ylabel('值')
            ax.set_title('极值范围')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 绘制分位数
            ax = axes[1, 0]
            quantile_cols = [col for col in feature_columns if col.startswith('quantile_')]
            if quantile_cols:
                for q_col in sorted(quantile_cols):
                    q_label = q_col.replace('quantile_', '') + '%'
                    ax.plot(x_data, column_df[q_col], label=q_label, linewidth=1.5, alpha=0.7)
            ax.set_xlabel(x_label)
            ax.set_ylabel('值')
            ax.set_title('分位数')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 绘制分布特征
            ax = axes[1, 1]
            plot_items = []
            if 'skewness' in feature_columns:
                line1 = ax.plot(x_data, column_df['skewness'], label='偏度', linewidth=2, color='orange')
                plot_items.append(line1[0])
            if 'kurtosis' in feature_columns:
                ax2 = ax.twinx()
                line2 = ax2.plot(x_data, column_df['kurtosis'], label='峰度', linewidth=2, color='green')
                plot_items.append(line2[0])
                ax2.set_ylabel('峰度', color='green')
                ax2.tick_params(axis='y', labelcolor='green')
            ax.set_xlabel(x_label)
            ax.set_ylabel('偏度', color='orange')
            ax.tick_params(axis='y', labelcolor='orange')
            ax.set_title('分布特征（偏度和峰度）')
            ax.legend(plot_items, [item.get_label() for item in plot_items], loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            file_path = os.path.join(output_path, f'{column_name}_features.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_paths.append(file_path)
    
    else:
        # 没有column列，绘制所有特征的组合图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('特征可视化', fontsize=16, fontweight='bold')
        
        # 使用索引作为x轴
        x_data = df.index
        
        # 绘制前几个主要特征
        main_features = [col for col in ['mean', 'std', 'min', 'max'] if col in feature_columns][:4]
        
        for idx, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            ax = axes[row, col]
            if idx < len(main_features):
                feature = main_features[idx]
                ax.plot(x_data, df[feature], label=feature, linewidth=2)
                ax.set_xlabel('窗口索引')
                ax.set_ylabel('值')
                ax.set_title(f'{feature} 特征')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        file_path = os.path.join(output_path, 'features_visualization.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(file_path)
    
    return saved_paths

