import os
import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Trainer:
    """模型训练器类"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[LRScheduler] = None,
        config: Any = None,
        device: str = "cuda"
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            optimizer: 优化器
            criterion: 损失函数
            scheduler: 学习率调度器
            config: 配置对象
            device: 设备
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # 混合精度训练
        self.scaler = GradScaler() if config.USE_AMP else None
        
        # 记录最佳验证指标
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
        # 早停计数器
        self.early_stopping_counter = 0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            metrics: 训练指标字典
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.config.USE_AMP:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
            
            # 计算指标
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
        # 计算平均指标
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
            
        return {
            'loss': avg_loss,
            'acc': accuracy
        }
        
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            metrics: 验证指标字典
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc='Validating')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 计算指标
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
        # 计算平均指标
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'acc': accuracy
        }
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            history: 训练历史记录
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(self.config.EPOCHS):
            logger.info(f"\nEpoch {epoch+1}/{self.config.EPOCHS}")
            
            # 训练一个epoch
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['acc'])
            
            # 验证
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['acc'])
            
            # 打印指标
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train Acc: {train_metrics['acc']:.2f}% "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Val Acc: {val_metrics['acc']:.2f}%"
            )
            
            # 保存最佳模型
            if val_metrics['acc'] > self.best_val_acc:
                logger.info(f"Validation accuracy improved from {self.best_val_acc:.2f} to {val_metrics['acc']:.2f}")
                self.best_val_acc = val_metrics['acc']
                self.save_checkpoint(self.config.BEST_MODEL_PATH)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                
            # 早停
            if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
                
            # 保存最后一个checkpoint
            self.save_checkpoint(self.config.LAST_MODEL_PATH)
            
        return history
        
    def save_checkpoint(self, path: str) -> None:
        """保存模型检查点
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str) -> None:
        """加载模型检查点
        
        Args:
            path: 检查点路径
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint {path} does not exist")
            return
            
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        logger.info(f"Checkpoint loaded from {path}") 