import os
import random
import shutil
from typing import Dict, List, Tuple
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

class DatasetSampler:
    """数据集采样工具类"""
    
    def __init__(self, config):
        """
        初始化采样器
        
        Args:
            config: 配置对象
        """
        self.config = config
        random.seed(config.RANDOM_SEED)
        
    def collect_samples(self, source_dir: str) -> Dict[str, List[str]]:
        """收集源目录中的所有样本
        
        Args:
            source_dir: 源数据目录
            
        Returns:
            Dict[str, List[str]]: 类别到样本路径的映射
        """
        samples = defaultdict(list)
        for category in os.listdir(source_dir):
            category_path = os.path.join(source_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples[category].append(os.path.join(category_path, filename))
                    
        return samples
        
    def sample_dataset(self, samples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """根据配置采样数据集
        
        Args:
            samples: 类别到样本路径的映射
            
        Returns:
            Dict[str, List[str]]: 采样后的类别到样本路径的映射
        """
        sampled = {}
        
        for category, paths in samples.items():
            if self.config.DATASET_SAMPLE_MODE == "percentage":
                sample_size = max(
                    int(len(paths) * self.config.DATASET_SAMPLE_SIZE),
                    self.config.MIN_SAMPLES_PER_CLASS
                )
            else:  # fixed_size mode
                sample_size = min(
                    int(self.config.DATASET_SAMPLE_SIZE),
                    len(paths)
                )
                
            # 确保至少有最小数量的样本
            sample_size = max(sample_size, self.config.MIN_SAMPLES_PER_CLASS)
            # 确保不超过可用样本数
            sample_size = min(sample_size, len(paths))
            
            sampled[category] = random.sample(paths, sample_size)
            
        return sampled
        
    def split_dataset(self, samples: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """将数据集分割为训练集和验证集
        
        Args:
            samples: 类别到样本路径的映射
            
        Returns:
            Tuple[Dict[str, List[str]], Dict[str, List[str]]]: 训练集和验证集的映射
        """
        train_samples = {}
        val_samples = {}
        
        for category, paths in samples.items():
            # 确保每个类别至少有最小数量的样本用于训练和验证
            min_samples = max(2 * self.config.MIN_SAMPLES_PER_CLASS, len(paths))
            if len(paths) < min_samples:
                logger.warning(
                    f"Category {category} has only {len(paths)} samples, "
                    f"which is less than the minimum required {min_samples}"
                )
            
            # 随机打乱样本
            random.shuffle(paths)
            
            # 计算分割点
            split_idx = int(len(paths) * (1 - self.config.TRAIN_VAL_SPLIT))
            
            train_samples[category] = paths[:split_idx]
            val_samples[category] = paths[split_idx:]
            
        return train_samples, val_samples
        
    def copy_samples(self, samples: Dict[str, List[str]], target_dir: str):
        """将样本复制到目标目录
        
        Args:
            samples: 类别到样本路径的映射
            target_dir: 目标目录
        """
        for category, paths in samples.items():
            category_dir = os.path.join(target_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            for src_path in paths:
                filename = os.path.basename(src_path)
                dst_path = os.path.join(category_dir, filename)
                shutil.copy2(src_path, dst_path)
                
    def process_dataset(self, source_dir: str, train_dir: str, val_dir: str):
        """处理数据集：采样和分割
        
        Args:
            source_dir: 源数据目录
            train_dir: 训练集目录
            val_dir: 验证集目录
        """
        logger.info("Collecting samples...")
        samples = self.collect_samples(source_dir)
        
        logger.info("Sampling dataset...")
        sampled = self.sample_dataset(samples)
        
        logger.info("Splitting dataset...")
        train_samples, val_samples = self.split_dataset(sampled)
        
        # 创建目标目录
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        logger.info("Copying training samples...")
        self.copy_samples(train_samples, train_dir)
        
        logger.info("Copying validation samples...")
        self.copy_samples(val_samples, val_dir)
        
        # 打印数据集统计信息
        self._print_stats(train_samples, val_samples)
        
    def _print_stats(self, train_samples: Dict[str, List[str]], val_samples: Dict[str, List[str]]):
        """打印数据集统计信息"""
        total_train = sum(len(paths) for paths in train_samples.values())
        total_val = sum(len(paths) for paths in val_samples.values())
        
        logger.info("\nDataset Statistics:")
        logger.info("-" * 50)
        logger.info(f"Total training samples: {total_train}")
        logger.info(f"Total validation samples: {total_val}")
        logger.info("\nPer-category statistics:")
        
        for category in train_samples.keys():
            train_count = len(train_samples[category])
            val_count = len(val_samples[category])
            logger.info(f"{category}:")
            logger.info(f"  Training: {train_count}")
            logger.info(f"  Validation: {val_count}")
            logger.info(f"  Total: {train_count + val_count}") 