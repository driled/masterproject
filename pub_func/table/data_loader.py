import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data(path):
    """
    加载QM7数据集并提取特征和目标变量

    参数:
    path: 包含CSV文件的目录路径

    返回:
    features: 分子特征数据
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

    # 明确指定header=0，确保第一行被识别为列名
    df = pd.read_csv(file_path, header=0)

    # 识别目标变量 - 可能与能量相关
    target_col = None
    potential_targets = ['energy', 'Energy', 'atomization_energy', 'atomization_E', 'E', 'ae_pbe0']  # 增加 ae_pbe0
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
        n_bins = 4  # 分成4个类别：低能量、中低能量、中高能量、高能量
        energy_bins = pd.qcut(energy_values, n_bins, labels=False)
        print(f"将能量分成 {n_bins} 个类别用于评估")
    else:
        energy_values = None
        energy_bins = None
        print("警告: 未找到与能量相关的列")

    # 移除非特征列
    non_feature_cols = []
    for col in df.columns:
        if ('name' in col.lower() or 'id' in col.lower() or 'smiles' in col.lower() or
                col == target_col or 'compound' in col.lower() or 'molecule' in col.lower()):
            non_feature_cols.append(col)

    features = df.drop(columns=non_feature_cols, errors='ignore')
    print(f"特征数据形状: {features.shape}")

    # 确保特征数据是数值型的
    try:
        features = features.astype(float)
    except ValueError as e:
        print(f"警告: 将特征转换为数值型时出错: {e}")
        print("尝试逐列转换...")
        for col in features.columns:
            try:
                features[col] = features[col].astype(float)
            except ValueError:
                print(f"无法将列 '{col}' 转换为数值型，删除该列")
                features = features.drop(columns=[col])

    print(f"清理后的特征数据形状: {features.shape}")
    return features, energy_values, energy_bins


def preprocess_data(X):
    """
    预处理特征数据（标准化和NaN处理）

    参数:
    X: 特征数据矩阵（DataFrame或ndarray）

    返回:
    X_scaled: 标准化的特征数据
    scaler: 用于后续变换的标准化器对象
    """
    # 检查X是否为DataFrame并处理NaN值
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        # 对于DataFrame或Series，使用pandas的isna方法
        if X.isna().any().any():  # 检查任何列中的任何单元格是否为NaN
            print(f"警告：数据中发现NaN值。应用插补...")
            # 将DataFrame转换为numpy数组以用于SimpleImputer
            X_array = X.values
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X_array)
            print(f"插补完成。形状保持不变：{X_imputed.shape}")
            X = X_imputed  # 使用插补后的数组
        else:
            # 如果没有NaN，转换为numpy数组用于后续处理
            X = X.values
    else:
        # 对于numpy数组，使用numpy的isnan方法
        if np.isnan(X).any():
            print(f"警告：数据中发现NaN值。应用插补...")
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            print(f"插补完成。形状保持不变：{X.shape}")

    # 确保X是numpy数组
    X = np.asarray(X)

    # 然后标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"数据标准化完成")
    return X_scaled, scaler