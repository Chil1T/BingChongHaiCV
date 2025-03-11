import os
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

class PlantDiseaseDataset(Dataset):
    """
    用于加载PlantVillage数据集的自定义Dataset类
    """
    def __init__(self, root_dir: str, image_size: int = 224, train: bool = True):
        """
        Args:
            root_dir (str): 数据集根目录路径
            image_size (int): 图像大小
            train (bool): 是否为训练模式
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.train = train
        
        # 设置transform
        self.transform = self._get_transform()
        
        # 获取所有类别
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 获取所有图片路径和标签
        self.samples = self._make_dataset()
        
        logger.info(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        logger.info(f"Classes: {self.classes}")
    
    def _get_transform(self) -> A.Compose:
        """获取数据增强转换"""
        if self.train:
            return A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def _make_dataset(self) -> List[Tuple[str, int]]:
        """构建数据集路径和标签列表"""
        samples = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root_dir, target_class)
            if not os.path.isdir(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, filename)
                    item = (path, self.class_to_idx[target_class])
                    samples.append(item)
                    
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本"""
        path, target = self.samples[idx]
        
        try:
            # 读取图片
            image = Image.open(path).convert('RGB')
            image = np.array(image)
            
            # 应用数据增强
            transformed = self.transform(image=image)
            image = transformed['image']
            
            return image, target
            
        except Exception as e:
            logger.error(f"Error loading image {path}: {str(e)}")
            # 返回数据集中的下一个有效样本
            return self.__getitem__((idx + 1) % len(self))
    
    def get_class_names(self):
        return self.classes

def create_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器
    
    Args:
        config: 配置对象
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 创建数据集
    train_dataset = PlantDiseaseDataset(
        root_dir=config.TRAIN_DIR,
        image_size=config.IMAGE_SIZE,
        train=True
    )
    
    val_dataset = PlantDiseaseDataset(
        root_dir=config.VAL_DIR,
        image_size=config.IMAGE_SIZE,
        train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader 