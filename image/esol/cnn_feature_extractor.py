"""
使用CNN提取图像特征的模块
"""

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from tqdm import tqdm


class FeatureExtractor:
    """
    使用预训练CNN模型提取图像特征
    """

    def __init__(self, model_name='resnet18', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化特征提取器

        参数:
        model_name: 使用的预训练模型名称，可选值: 'resnet18', 'resnet50', 'vgg16', 'densenet121', 'efficientnet'
        device: 使用的设备，'cuda'或'cpu'
        """
        self.device = device
        self.model_name = model_name
        print(f"使用 {model_name} 作为特征提取器，运行在 {device} 上")

        # 加载预训练模型
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])  # 移除最后的全连接层
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            self.model = nn.Sequential(*list(model.features), model.avgpool)
            self.feature_dim = 512 * 7 * 7
        elif model_name == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            self.model = nn.Sequential(*list(self.model.features), nn.AdaptiveAvgPool2d((1, 1)))
            self.feature_dim = 1024
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 1280
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")

        # 将模型移至指定设备并设置为评估模式
        self.model = self.model.to(device)
        self.model.eval()

    def extract_features(self, data_loader, flatten=True, normalize=False):
        """
        从数据加载器中提取特征

        参数:
        data_loader: 包含图像数据的PyTorch数据加载器
        flatten: 是否将特征展平为向量
        normalize: 是否对特征进行标准化

        返回:
        features: 提取的特征数组
        """
        features = []
        labels = []

        # 禁用梯度计算以加速推理
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="提取特征"):
                # 处理同时有图像和标签的情况
                if isinstance(batch, list) and len(batch) == 2:
                    images, batch_labels = batch
                    images = images.to(self.device)
                    labels.extend(batch_labels.numpy())
                else:
                    images = batch.to(self.device)

                # 前向传播
                outputs = self.model(images)

                # 处理不同模型的输出格式
                if self.model_name == 'vgg16':
                    outputs = outputs.view(outputs.size(0), -1)
                else:
                    outputs = outputs.squeeze()

                # 将特征移至CPU并转换为NumPy数组
                outputs = outputs.cpu().numpy()

                # 添加到特征列表
                features.append(outputs)

        # 合并所有批次的特征
        features = np.vstack(features)

        # 如果需要展平特征
        if flatten and len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)

        # 如果需要标准化特征
        if normalize:
            # 减去均值并除以标准差
            features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

        print(f"提取的特征形状: {features.shape}")

        if len(labels) > 0:
            return features, np.array(labels)

        return features

    def get_feature_dim(self):
        """
        获取特征维度

        返回:
        feature_dim: 特征维度
        """
        return self.feature_dim


def extract_features_from_images(image_files, model_name='resnet18', batch_size=32):
    """
    从图像文件中提取特征的便捷函数

    参数:
    image_files: 图像文件路径列表
    solubility_values: 溶解度值（可选）
    model_name: 使用的CNN模型名称
    batch_size: 批次大小

    返回:
    features: 提取的特征数组
    labels: 溶解度值数组（如果提供）
    """
    from image_data_loader import create_data_loader, MoleculeImageDataset

    # 创建数据加载器
    data_loader = create_data_loader(image_files,  batch_size=batch_size)

    # 创建特征提取器
    extractor = FeatureExtractor(model_name=model_name)

    # 提取特征
    #
    features = extractor.extract_features(data_loader, normalize=True)
    return features