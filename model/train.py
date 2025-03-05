import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import logging
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .dataloader.dataset import PlantDiseaseDataset
from .config.train_config import TrainingConfig as cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 根据是否使用混合精度训练选择不同的训练流程
        if scaler is not None:  # 使用混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # 普通训练
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{running_loss/len(train_loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{running_loss/len(val_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss/len(val_loader), 100.*correct/total

def main():
    # 创建保存目录
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    
    # 设置设备
    device = torch.device(cfg.DEVICE)
    logger.info(f"Using device: {device}")
    
    # 数据增强
    train_transform = A.Compose([
        # 1. 空间变换
        A.RandomResizedCrop(
            height=cfg.IMAGE_SIZE, 
            width=cfg.IMAGE_SIZE,
            scale=(0.8, 1.0),  # 保持较大的裁剪比例，避免丢失病害特征
            ratio=(0.9, 1.1)   # 略微改变长宽比
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),  # 部分植物病害图片垂直翻转也合理
        A.Rotate(limit=15, p=0.5),  # 小角度旋转
        
        # 2. 颜色和光照变换
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.8),
        
        # 3. 天气和环境模拟
        A.OneOf([
            A.RandomShadow(p=1.0),
            A.RandomSunFlare(p=1.0),
            A.RandomFog(p=1.0),
            A.RandomRain(p=1.0)
        ], p=0.3),
        
        # 4. 图像质量变换
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0)
        ], p=0.2),
        
        # 5. 局部变换
        A.OneOf([
            A.CoarseDropout(
                max_holes=8,
                max_height=cfg.IMAGE_SIZE//16,
                max_width=cfg.IMAGE_SIZE//16,
                min_holes=5,
                fill_value=0,
                p=1.0
            ),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(p=1.0),
        ], p=0.2),
        
        # 6. 标准化
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # 验证集只需要调整大小和标准化
    val_transform = A.Compose([
        A.Resize(height=cfg.IMAGE_SIZE, width=cfg.IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # 数据加载
    train_dataset = PlantDiseaseDataset(root_dir=cfg.TRAIN_DIR, transform=train_transform)
    val_dataset = PlantDiseaseDataset(root_dir=cfg.VAL_DIR, transform=val_transform)
    
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
    
    # 更新类别数
    cfg.NUM_CLASSES = len(train_dataset.classes)
    logger.info(f"Number of classes: {cfg.NUM_CLASSES}")
    logger.info(f"Class names: {train_dataset.classes}")
    
    # 初始化模型
    model = models.resnet50(pretrained=cfg.PRETRAINED)
    model.fc = nn.Linear(model.fc.in_features, cfg.NUM_CLASSES)
    model = model.to(device)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = AdamW(
        model.parameters(),
        **cfg.OPTIMIZER['params']
    )
    
    # 定义学习率调度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        **cfg.SCHEDULER['params']
    )
    
    # 初始化梯度缩放器（仅在使用GPU且启用AMP时使用）
    scaler = GradScaler(enabled=cfg.USE_AMP) if torch.cuda.is_available() and cfg.USE_AMP else None
    
    # 早停
    early_stopping = EarlyStopping(
        patience=cfg.EARLY_STOPPING_PATIENCE,
        min_delta=cfg.EARLY_STOPPING_MIN_DELTA
    )
    
    # 训练循环
    best_val_acc = 0
    for epoch in range(cfg.EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{cfg.EPOCHS}")
        
        # 训练阶段
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, scaler, device
        )
        logger.info(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'val_acc': val_acc,
                'classes': train_dataset.classes
            }, cfg.BEST_MODEL_PATH)
            logger.info(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main() 