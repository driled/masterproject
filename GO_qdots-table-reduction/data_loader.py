import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data(path):
    """
    加载CSV数据并排除ID和file_name列

    参数:
    path: 包含CSV文件的目录路径

    返回:
    features: 只包含数值特征的数据
    energy_values: 连续能量值（用于可视化）
    energy_bins: 离散化能量类别（用于评估聚类质量）
    """
    # 在目录中查找CSV文件
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"在目录 {path} 中没有找到CSV文件")

    # 加载找到的第一个CSV文件
    file_path = os.path.join(path, files[0])
    print(f"从 {file_path} 加载数据")

    # 读取CSV文件，明确排除前两列
    df = pd.read_csv(file_path)

    # 移除ID和file_name列
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    if 'file_name' in df.columns:
        df = df.drop(columns=['file_name'])

    print(f"排除ID和file_name列后的数据形状: {df.shape}")

    # 识别目标变量 - 可能与能量相关
    target_col = None
    potential_targets = ['Formation_energy', 'Fermi_energy', 'energy', 'Energy']
    for col in potential_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col:
        # 保存连续能量值（用于可视化）
        energy_values = df[target_col]
        print(f"目标变量: {target_col}")

        # 创建离散化能量类别（用于评估聚类质量）
        # 使用分位数将连续值分成类别
        n_bins = 4  # 分成4个类别
        energy_bins = pd.qcut(energy_values, n_bins, labels=False)
        print(f"将能量分成 {n_bins} 个类别用于评估")
    else:
        energy_values = None
        energy_bins = None
        print("警告: 未找到与能量相关的列")

    # 确保所有特征都是数值型
    features = df

    for col in features.columns:
        if features[col].dtype == 'object':
            print(f"警告: 列 '{col}' 包含非数值数据，尝试转换...")
            try:
                features[col] = pd.to_numeric(features[col], errors='coerce')
            except:
                print(f"无法将列 '{col}' 转换为数值型，将用NaN填充")
                features[col] = np.nan

    print(f"最终数值特征数据形状: {features.shape}")
    return features, energy_values, energy_bins


def preprocess_data(X):
    """
    预处理特征数据（标准化和NaN处理）

    参数:
    X: 特征数据矩阵

    返回:
    X_scaled: 标准化的特征数据
    scaler: 用于后续变换的标准化器对象
    """
    # 首先，处理NaN值
    if np.isnan(X.values).any():
        print(f"警告：数据中发现NaN值。应用插补...")
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        print(f"插补完成。形状保持不变：{X.shape} -> {X_imputed.shape}")
        X_processed = X_imputed
    else:
        X_processed = X.values

    # 然后标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    print(f"数据标准化完成")
    return X_scaled, scaler