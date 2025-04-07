"""
加载ESOL分子图像表示的数据加载模块
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob


def load_image_data(image_dir, metadata_path=None):
    """
    加载ESOL图像数据和对应的元数据

    参数:
    image_dir: 包含分子图像的目录路径
    metadata_path: 包含溶解度数据的CSV文件路径（如果有）

    返回:
    images_list: 图像文件路径列表
    solubility_values: 连续的溶解度值（用于可视化）
    solubility_bins: 离散化的溶解度类别（用于评估聚类质量）
    molecule_ids: 分子ID列表
    """
    print(f"正在从 {image_dir} 加载图像数据")

    # 查找所有图像文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_files:
        raise FileNotFoundError(f"目录 {image_dir} 中未找到图像文件")

    print(f"找到 {len(image_files)} 个图像文件")

    # 从文件名提取分子ID
    molecule_ids = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

    # 如果有元数据文件，则加载溶解度信息
    solubility_values = None
    solubility_bins = None

    if metadata_path and os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        print(f"加载元数据: {metadata_path}")

        # 识别目标变量 - 可能是 'Solubility' 或 'measured log solubility in mols per litre'
        target_col = None
        id_col = None

        # 尝试找到ID列
        for col in df.columns:
            if 'id' in col.lower() or 'name' in col.lower() or 'molecule' in col.lower():
                id_col = col
                break

        # 尝试找到溶解度列
        if 'Solubility' in df.columns:
            target_col = 'Solubility'
        elif 'measured log solubility in mols per litre' in df.columns:
            target_col = 'measured log solubility in mols per litre'

        if target_col and id_col:
            # 创建分子ID到溶解度的映射
            id_to_solubility = dict(zip(df[id_col], df[target_col]))

            # 按图像文件的分子ID顺序获取溶解度值
            solubility_values = np.array([id_to_solubility.get(mid, np.nan) for mid in molecule_ids])

            # 删除缺失的溶解度值
            valid_indices = ~np.isnan(solubility_values)
            if not all(valid_indices):
                print(f"警告: {sum(~valid_indices)} 个分子没有溶解度数据")
                image_files = [f for i, f in enumerate(image_files) if valid_indices[i]]
                molecule_ids = [m for i, m in enumerate(molecule_ids) if valid_indices[i]]
                solubility_values = solubility_values[valid_indices]

            print(f"目标变量: {target_col}")

            # 创建离散化的溶解度类别（用于评估聚类质量）
            # 使用分位数将连续值分成多个类别
            n_bins = 4  # 分成4个类别：低溶解度，中低溶解度，中高溶解度，高溶解度
            solubility_bins = pd.qcut(solubility_values, n_bins, labels=False)
            print(f"将溶解度分成 {n_bins} 个类别用于评估")
        else:
            print("警告: 在元数据中未找到溶解度或分子ID列")

    return image_files, solubility_values, solubility_bins, molecule_ids


class MoleculeImageDataset(Dataset):
    """
    分子图像数据集类，用于加载和预处理分子图像
    """

    def __init__(self, image_files, labels=None, transform=None):
        """
        初始化数据集

        参数:
        image_files: 图像文件路径列表
        labels: 标签（溶解度）列表，可以为None
        transform: 图像变换操作
        """
        self.image_files = image_files
        self.labels = labels

        # 如果没有提供变换，使用默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整大小为标准尺寸
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')  # 确保为RGB格式

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 如果有标签，则返回图像和标签
        if self.labels is not None:
            return image, self.labels[idx]

        # 否则只返回图像
        return image


def create_data_loader(image_files, solubility_values=None, batch_size=32, num_workers=4):
    """
    创建数据加载器

    参数:
    image_files: 图像文件路径列表
    solubility_values: 溶解度值列表（可选）
    batch_size: 批次大小
    num_workers: 工作线程数

    返回:
    data_loader: PyTorch数据加载器
    """
    # 创建数据集
    dataset = MoleculeImageDataset(image_files, solubility_values)

    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return data_loader