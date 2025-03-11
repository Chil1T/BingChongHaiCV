from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.models as models

class ModelFactory:
    """模型工厂类，用于创建模型、优化器、损失函数和学习率调度器"""
    
    @staticmethod
    def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
        """创建模型
        
        Args:
            model_name: 模型名称
            num_classes: 类别数
            pretrained: 是否使用预训练权重
            
        Returns:
            model: 创建的模型
        """
        # 获取模型创建函数
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            raise ValueError(f"Model {model_name} not found in torchvision.models")
            
        # 创建模型
        model = model_fn(pretrained=pretrained)
        
        # 修改最后一层
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
                
        return model
        
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optimizer_name: str,
        lr: float,
        weight_decay: float,
        **kwargs
    ) -> optim.Optimizer:
        """创建优化器
        
        Args:
            model: 模型
            optimizer_name: 优化器名称
            lr: 学习率
            weight_decay: 权重衰减
            **kwargs: 其他参数
            
        Returns:
            optimizer: 创建的优化器
        """
        # 获取优化器类
        optimizer_cls = getattr(optim, optimizer_name, None)
        if optimizer_cls is None:
            raise ValueError(f"Optimizer {optimizer_name} not found in torch.optim")
            
        # 创建优化器
        return optimizer_cls(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_name: str,
        **kwargs
    ) -> Optional[_LRScheduler]:
        """创建学习率调度器
        
        Args:
            optimizer: 优化器
            scheduler_name: 调度器名称
            **kwargs: 其他参数
            
        Returns:
            scheduler: 创建的调度器
        """
        if not scheduler_name:
            return None
            
        # 获取调度器类
        scheduler_cls = getattr(optim.lr_scheduler, scheduler_name, None)
        if scheduler_cls is None:
            raise ValueError(f"Scheduler {scheduler_name} not found in torch.optim.lr_scheduler")
            
        # 创建调度器
        return scheduler_cls(optimizer, **kwargs)
        
    @staticmethod
    def create_criterion() -> nn.Module:
        """创建损失函数
        
        Returns:
            criterion: 创建的损失函数
        """
        return nn.CrossEntropyLoss()
        
    @classmethod
    def create_training_components(
        cls,
        config: Any
    ) -> Tuple[nn.Module, optim.Optimizer, nn.Module, Optional[_LRScheduler]]:
        """创建训练所需的所有组件
        
        Args:
            config: 配置对象
            
        Returns:
            model: 模型
            optimizer: 优化器
            criterion: 损失函数
            scheduler: 学习率调度器
        """
        try:
            # 创建模型
            model_name = getattr(config, 'MODEL_NAME', 'resnet50')
            num_classes = getattr(config, 'NUM_CLASSES', 38)
            pretrained = getattr(config, 'PRETRAINED', True)
            
            model = cls.create_model(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained
            )
            
            # 创建优化器
            optimizer_name = 'AdamW'  # 默认值
            
            if hasattr(config, 'OPTIMIZER'):
                optimizer_config = getattr(config, 'OPTIMIZER')
                if isinstance(optimizer_config, dict):
                    optimizer_name = optimizer_config.get('name', optimizer_name)
                    
                    if 'params' in optimizer_config:
                        # 直接使用配置中的参数创建优化器
                        optimizer_params = optimizer_config['params']
                        optimizer_cls = getattr(optim, optimizer_name, None)
                        if optimizer_cls is None:
                            raise ValueError(f"Optimizer {optimizer_name} not found in torch.optim")
                        
                        optimizer = optimizer_cls(model.parameters(), **optimizer_params)
                    else:
                        # 如果没有params字段，使用默认参数
                        lr = getattr(config, 'LEARNING_RATE', 0.001)
                        weight_decay = getattr(config, 'WEIGHT_DECAY', 1e-4)
                        
                        optimizer = cls.create_optimizer(
                            model=model,
                            optimizer_name=optimizer_name,
                            lr=lr,
                            weight_decay=weight_decay
                        )
            else:
                # 使用默认参数
                lr = getattr(config, 'LEARNING_RATE', 0.001)
                weight_decay = getattr(config, 'WEIGHT_DECAY', 1e-4)
                
                optimizer = cls.create_optimizer(
                    model=model,
                    optimizer_name=optimizer_name,
                    lr=lr,
                    weight_decay=weight_decay
                )
            
            # 创建损失函数
            criterion = cls.create_criterion()
            
            # 创建学习率调度器
            scheduler_name = ''
            scheduler_params = {}
            
            if hasattr(config, 'SCHEDULER'):
                scheduler_config = getattr(config, 'SCHEDULER')
                if isinstance(scheduler_config, dict):
                    scheduler_name = scheduler_config.get('name', '')
                    if 'params' in scheduler_config:
                        scheduler_params = scheduler_config['params']
            
            scheduler = cls.create_scheduler(
                optimizer=optimizer,
                scheduler_name=scheduler_name,
                **scheduler_params
            )
            
            return model, optimizer, criterion, scheduler
        
        except Exception as e:
            import traceback
            print(f"Error in create_training_components: {str(e)}")
            print(traceback.format_exc())
            raise 