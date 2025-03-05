import os
import sys
import logging
from pathlib import Path
import shutil
import random
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.dataloader.dataset import PlantDiseaseDataset
from model.train import train_epoch, validate
from model.config.train_config import TrainingConfig as cfg

def create_small_dataset(source_dir, target_dir, samples_per_class=10):
    """创建小规模数据集用于验证"""
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历每个类别
    for category in os.listdir(source_dir):
        source_category_dir = os.path.join(source_dir, category)
        target_category_dir = os.path.join(target_dir, category)
        
        if os.path.isdir(source_category_dir):
            os.makedirs(target_category_dir, exist_ok=True)
            
            # 获取该类别下的所有图片
            images = [f for f in os.listdir(source_category_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 随机选择指定数量的图片
            selected_images = random.sample(images, min(samples_per_class, len(images)))
            
            # 复制选中的图片
            for img in tqdm(selected_images, desc=f"Copying {category} images", leave=False):
                src = os.path.join(source_category_dir, img)
                dst = os.path.join(target_category_dir, img)
                shutil.copy2(src, dst)

def validate_training():
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # 创建小规模数据集目录
    small_data_dir = os.path.join(project_root, "data_small")
    small_train_dir = os.path.join(small_data_dir, "train")
    small_val_dir = os.path.join(small_data_dir, "val")
    
    try:
        # 从原始数据集创建小规模数据集
        logger.info("Creating small dataset for validation...")
        create_small_dataset(cfg.TRAIN_DIR, small_train_dir, samples_per_class=10)
        create_small_dataset(cfg.VAL_DIR, small_val_dir, samples_per_class=5)
        
        # 修改配置以适应CPU训练
        cfg.BATCH_SIZE = 8  # 减小batch size
        cfg.NUM_WORKERS = 0  # CPU模式下不使用多进程
        cfg.PIN_MEMORY = False  # CPU模式下不需要
        cfg.USE_AMP = False  # CPU模式下不使用混合精度训练
        cfg.EPOCHS = 2  # 仅训练几个epoch用于验证
        
        # 创建数据加载器
        train_dataset = PlantDiseaseDataset(root_dir=small_train_dir)
        val_dataset = PlantDiseaseDataset(root_dir=small_val_dir)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )
        
        # 初始化模型和训练组件
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.classes))
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), **cfg.OPTIMIZER['params'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **cfg.SCHEDULER['params'])
        
        # 记录开始时间
        start_time = time.time()
        
        # 进行训练
        logger.info("Starting validation training...")
        for epoch in range(cfg.EPOCHS):
            logger.info(f"\nEpoch {epoch+1}/{cfg.EPOCHS}")
            
            # 由于我们在CPU上训练，不使用梯度缩放器
            train_loss, train_acc = train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,  # CPU训练不需要梯度缩放器
                device=cfg.DEVICE
            )
            
            val_loss, val_acc = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=cfg.DEVICE
            )
            
            logger.info(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # 记录总用时
        total_time = time.time() - start_time
        logger.info(f"\nValidation training completed in {total_time:.2f} seconds!")
        
        # 清理临时数据集
        shutil.rmtree(small_data_dir)
        logger.info("Cleaned up temporary dataset")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        if os.path.exists(small_data_dir):
            shutil.rmtree(small_data_dir)
            logger.info("Cleaned up temporary dataset after error")
        return False

if __name__ == "__main__":
    if validate_training():
        print("\nValidation successful! You can proceed with cloud training.")
    else:
        print("\nValidation failed. Please check the errors above.") 