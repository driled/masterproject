import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    """
    加载ESOL数据集并提取特征和目标变量

    参数:
    path: 包含CSV文件的目录路径

    返回:
    features: 分子特征数据
    solubility_values: 连续的溶解度值（用于可视化）
    solubility_bins: 离散化的溶解度类别（用于评估聚类质量）
    """
    # 查找目录中的CSV文件
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"目录 {path} 中未找到CSV文件")

    # 加载第一个找到的CSV文件
    file_path = os.path.join(path, files[0])
    print(f"正在从 {file_path} 加载数据")

    df = pd.read_csv(file_path)

    # 识别目标变量 - 可能是 'Solubility' 或 'measured log solubility in mols per litre'
    target_col = None
    if 'Solubility' in df.columns:
        target_col = 'Solubility'
    elif 'measured log solubility in mols per litre' in df.columns:
        target_col = 'measured log solubility in mols per litre'

    if target_col:
        # 保存连续型溶解度值（用于可视化）
        solubility_values = df[target_col]
        print(f"目标变量: {target_col}")

        # 创建离散化的溶解度类别（用于评估聚类质量）
        # 使用分位数将连续值分成多个类别
        n_bins = 4  # 分成4个类别：低溶解度，中低溶解度，中高溶解度，高溶解度
        solubility_bins = pd.qcut(solubility_values, n_bins, labels=False)
        print(f"将溶解度分成 {n_bins} 个类别用于评估")
    else:
        solubility_values = None
        solubility_bins = None
        print("警告: 没有找到溶解度相关的列")

    # 移除非特征列
    non_feature_cols = []
    for col in df.columns:
        if ('name' in col.lower() or 'id' in col.lower() or 'smiles' in col.lower() or
                col == target_col or 'compound' in col.lower()):
            non_feature_cols.append(col)

    features = df.drop(columns=non_feature_cols, errors='ignore')
    print(f"特征数据形状: {features.shape}")

    return features, solubility_values, solubility_bins


def preprocess_data(X):
    """
    对特征数据进行预处理（标准化）

    参数:
    X: 特征数据矩阵

    返回:
    X_scaled: 标准化后的特征数据
    scaler: 标准化器对象，用于后续转换
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"数据标准化完成")
    return X_scaled, scaler