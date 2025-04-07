import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def load_image_data(image_dir):
    """
    加载图像文件路径列表（支持 png/jpg/jpeg）

    参数:
    image_dir: 图像文件所在目录

    返回:
    image_files: 图像文件路径列表
    molecule_ids: 图像文件对应的分子ID列表（从文件名提取）
    """
    print(f"正在从 {image_dir} 加载图像数据...")

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_files:
        raise FileNotFoundError(f"目录 {image_dir} 中未找到图像文件")

    print(f"找到 {len(image_files)} 个图像文件")
    molecule_ids = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

    return image_files, molecule_ids


class MoleculeImageDataset(Dataset):
    """
    图像数据集类，不含标签，仅用于特征提取
    """
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image


def create_data_loader(image_files, batch_size=32, num_workers=4):
    """
    创建仅加载图像的 PyTorch 数据加载器

    参数:
    image_files: 图像路径列表
    返回:
    data_loader: 无标签的图像加载器
    """
    dataset = MoleculeImageDataset(image_files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader
