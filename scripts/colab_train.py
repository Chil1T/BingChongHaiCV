import os
import sys
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.dataloader.dataset import PlantDiseaseDataset
from model.config.colab_config import ColabTrainingConfig as cfg

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(cfg.LOG_DIR, 'training.log'))
    ]
)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.lrs = []
        
    def update(self, train_loss, train_acc, val_loss, val_acc, lr):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.lrs.append(lr)
        
    def plot(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        # 创建一个2x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, self.train_accs, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 学习率曲线
        ax3.plot(epochs, self.lrs, 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # 训练vs验证准确率对比
        ax4.scatter(self.train_accs, self.val_accs, alpha=0.5)
        ax4.plot([0, 100], [0, 100], 'r--')  # 对角线
        ax4.set_title('Training vs Validation Accuracy')
        ax4.set_xlabel('Training Accuracy (%)')
        ax4.set_ylabel('Validation Accuracy (%)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 保存图表
        fig.savefig(os.path.join(cfg.LOG_DIR, 'training_curves.png'))
        
        # 显示最终指标
        metrics_df = pd.DataFrame({
            'Metric': ['Best Training Loss', 'Best Training Accuracy', 
                      'Best Validation Loss', 'Best Validation Accuracy'],
            'Value': [min(self.train_losses), max(self.train_accs),
                     min(self.val_losses), max(self.val_accs)]
        })
        display(metrics_df)

def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 使用混合精度训练
        with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # 使用梯度缩放器
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
    # 创建必要的目录
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    
    # 设置设备
    device = torch.device(cfg.DEVICE)
    logger.info(f"Using device: {device}")
    
    # 创建数据加载器
    train_dataset = PlantDiseaseDataset(root_dir=cfg.TRAIN_DIR)
    val_dataset = PlantDiseaseDataset(root_dir=cfg.VAL_DIR)
    
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
    model = torch.hub.load('pytorch/vision:v0.10.0', cfg.MODEL_NAME, pretrained=cfg.PRETRAINED)
    model.fc = nn.Linear(model.fc.in_features, cfg.NUM_CLASSES)
    model = model.to(device)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **cfg.OPTIMIZER['params']
    )
    
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        **cfg.SCHEDULER['params']
    )
    
    # 初始化梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP)
    
    # 初始化训练监视器
    monitor = TrainingMonitor()
    
    # 训练循环
    best_val_acc = 0
    start_time = time.time()
    
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
        
        # 更新监视器
        monitor.update(
            train_loss, train_acc,
            val_loss, val_acc,
            scheduler.get_last_lr()[0]
        )
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': val_acc,
                'classes': train_dataset.classes
            }, cfg.BEST_MODEL_PATH)
            logger.info(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        
        # 保存最新模型（用于断点续训）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'monitor_data': {
                'train_losses': monitor.train_losses,
                'train_accs': monitor.train_accs,
                'val_losses': monitor.val_losses,
                'val_accs': monitor.val_accs,
                'lrs': monitor.lrs
            }
        }, cfg.LAST_MODEL_PATH)
    
    # 训练结束
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time/3600:.2f} hours!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # 绘制训练曲线
    monitor.plot()

if __name__ == '__main__':
    main() 