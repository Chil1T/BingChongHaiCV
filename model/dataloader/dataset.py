import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

class PlantDiseaseDataset(Dataset):
    """
    用于加载PlantVillage数据集的自定义Dataset类
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 数据集根目录路径
            transform (callable, optional): 可选的图像转换
        """
        self.root_dir = root_dir
        self.transform = transform or A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        # 获取所有类别（文件夹名称）
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 收集所有图像路径和对应的标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        logger.info(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        logger.info(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (image, label) 其中image是转换后的图像，label是对应的类别索引
        """
        img_path, label = self.samples[idx]
        
        try:
            # 使用PIL加载图像
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # 应用转换
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # 返回数据集中的下一个有效样本
            return self.__getitem__((idx + 1) % len(self))
    
    def get_class_names(self):
        return self.classes 