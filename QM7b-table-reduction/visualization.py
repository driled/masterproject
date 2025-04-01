import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


def handle_nan_before_visualization(X, values=None):
    """
    检查并处理可视化前的NaN值

    参数:
    X: 用于可视化的数据
    values: 用于着色的值

    返回:
    X_clean: 不含NaN值的数据
    values_clean: 不含NaN条目的值
    removed_indices: 被移除点的索引
    had_nan: 布尔值，表示是否存在NaN值
    """
    # 检查X中的NaN值
    nan_mask_X = np.isnan(X).any(axis=1)

    # 如果提供了values，检查其中的NaN值
    if values is not None:
        if isinstance(values, np.ndarray):
            nan_mask_values = np.isnan(values)
        else:
            # 处理pandas Series或其他类型
            nan_mask_values = pd.isna(values).values

        # 合并掩码
        nan_mask = nan_mask_X | nan_mask_values
    else:
        nan_mask = nan_mask_X

    had_nan = np.any(nan_mask)

    if had_nan:
        # 获取要删除的点的索引
        removed_indices = np.where(nan_mask)[0]
        kept_indices = np.where(~nan_mask)[0]

        print(f"警告: 为了可视化，移除了 {np.sum(nan_mask)} 个带有NaN值的点")

        # 过滤数据
        X_clean = X[~nan_mask]

        # 如果提供了values，过滤它们
        if values is not None:
            if isinstance(values, np.ndarray):
                values_clean = values[~nan_mask]
            else:
                values_clean = values.iloc[kept_indices]
        else:
            values_clean = None

        return X_clean, values_clean, removed_indices, had_nan

    # 如果没有NaN值，返回原始数据
    return X, values, [], False


def save_embedding_csv(X_embedded, output_path, filename):
    """
    将降维结果保存到CSV文件

    参数:
    X_embedded: 降维后的数据
    output_path: 输出目录路径
    filename: 输出文件名
    """
    # 处理可能存在的NaN值
    X_clean, _, removed_indices, had_nan = handle_nan_before_visualization(X_embedded)

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, filename)

    if had_nan:
        # 保存时包含NaN指示器列
        df = pd.DataFrame(X_embedded)
        df['had_nan'] = np.isnan(X_embedded).any(axis=1)
        df.to_csv(output_file, index=False)
        print(f"带有NaN指示器的降维结果已保存到: {output_file}")
    else:
        pd.DataFrame(X_embedded).to_csv(output_file, index=False)
        print(f"降维结果已保存到: {output_file}")


def plot_2d_embedding(X_embedded, values, output_path, filename, title, colormap='viridis', label='Value'):
    """
    将2D降维结果绘制为散点图

    参数:
    X_embedded: 2D降维数据
    values: 用于着色的值
    output_path: 输出目录路径
    filename: 输出文件名
    title: 图表标题
    colormap: 色图
    label: 颜色条标签
    """
    # 处理NaN值
    X_clean, values_clean, _, had_nan = handle_nan_before_visualization(X_embedded, values)

    if X_clean.shape[0] == 0:
        print(f"警告: 清除NaN值后没有剩余数据可绘制。已跳过绘制 {filename}")
        return

    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(X_clean[:, 0], X_clean[:, 1], c=values_clean, cmap=colormap, alpha=0.7)

    if had_nan:
        plt.title(f"{title}\n(注意: 移除了包含NaN值的点)")
    else:
        plt.title(title)

    plt.colorbar(scatter, label=label)

    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    if had_nan:
        print(f"图像（已移除NaN值）已保存到: {output_file}")
    else:
        print(f"图像已保存到: {output_file}")


def plot_clustering_result(X_embedded, cluster_labels, output_path, filename, title):
    """
    将聚类结果绘制为散点图

    参数:
    X_embedded: 2D降维数据
    cluster_labels: 聚类标签
    output_path: 输出目录路径
    filename: 输出文件名
    title: 图表标题
    """
    # 处理NaN值
    X_clean, labels_clean, _, had_nan = handle_nan_before_visualization(X_embedded, cluster_labels)

    if X_clean.shape[0] == 0:
        print(f"警告: 清除NaN值后没有剩余数据可绘制。已跳过绘制 {filename}")
        return

    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(X_clean[:, 0], X_clean[:, 1], c=labels_clean, cmap='tab10', alpha=0.7)

    if had_nan:
        plt.title(f"{title}\n(注意: 移除了包含NaN值的点)")
    else:
        plt.title(title)

    plt.colorbar(scatter, label='聚类')

    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    if had_nan:
        print(f"聚类结果图像（已移除NaN值）已保存到: {output_file}")
    else:
        print(f"聚类结果图像已保存到: {output_file}")


def plot_silhouette_history(history, best_silhouette, output_path, filename, title):
    """
    绘制轮廓系数与聚类数量的关系图

    参数:
    history: 包含聚类数量和轮廓系数的历史记录
    best_silhouette: 最佳轮廓系数
    output_path: 输出目录路径
    filename: 输出文件名
    title: 图表标题
    """
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(10, 6))

    plt.plot([item['n_clusters'] for item in history],
             [item['silhouette'] for item in history],
             marker='o')

    plt.axhline(y=best_silhouette, color='r', linestyle='--')
    plt.xlabel('聚类数量')
    plt.ylabel('轮廓系数')
    plt.title(title)
    plt.grid(True)

    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    print(f"轮廓历史图像已保存到: {output_file}")


def plot_comparison_bar(data, x_label, y_label, title, output_path, filename):
    """
    绘制比较条形图

    参数:
    data: 包含比较数据的DataFrame
    x_label: X轴标签
    y_label: Y轴标签
    title: 图表标题
    output_path: 输出目录路径
    filename: 输出文件名
    """
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(14, 7))

    # 获取唯一方法名称
    methods = data['Method'].unique()

    bar_width = 0.15
    index = np.arange(len(methods))
    param_groups = data.groupby(['Method', 'Parameters'])

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    # 为每个方法-参数组合创建条形图
    for method in methods:
        method_params = data[data['Method'] == method]['Parameters'].unique()
        for i, params in enumerate(method_params):
            subset = data[(data['Method'] == method) & (data['Parameters'] == params)]
            if len(subset) > 0 and y_label in subset.columns:
                # 确保有值可绘制
                if not pd.isna(subset[y_label].values[0]):
                    pos = index[np.where(methods == method)[0][0]] + (i - len(method_params) / 2 + 0.5) * bar_width
                    plt.bar(pos, subset[y_label].values[0], bar_width,
                            label=f"{method}: {params}", color=colors[color_idx % len(colors)])
            color_idx += 1

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(index, methods, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    print(f"比较图表已保存到: {output_file}")


def save_results_summary(results_df, output_path, filename):
    """
    将结果摘要保存为CSV文件

    参数:
    results_df: 包含结果的DataFrame
    output_path: 输出目录路径
    filename: 输出文件名
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, filename)
    results_df.to_csv(output_file, index=False)
    print(f"结果摘要已保存到: {output_file}")