from typing import Dict, Any
import torch
from .base_config import BaseConfig

class ColabConfig(BaseConfig):
    """Colab特化配置类"""
    
    # 路径配置
    DATA_ROOT = "/content/data"  # 修改为实际数据目录
    MODEL_DIR = "/content/BingChongHaiCV/models"
    LOG_DIR = "/content/BingChongHaiCV/logs"
    
    # 覆盖基础配置
    TRAIN_DIR = f"{DATA_ROOT}/train"
    VAL_DIR = f"{DATA_ROOT}/val"
    BATCH_SIZE = 32  # 降低batch size以适应内存
    NUM_WORKERS = 4  # 增加worker数量提高效率
    EPOCHS = 50  # 增加训练轮数
    
    # 检查点配置
    CHECKPOINT_DIR = MODEL_DIR
    BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pth"
    LAST_MODEL_PATH = f"{CHECKPOINT_DIR}/last_model.pth"
    
    # 数据集配置
    IMAGE_SIZE = 224
    PIN_MEMORY = True
    
    # 模型配置
    MODEL_NAME = "resnet50"
    PRETRAINED = True
    NUM_CLASSES = 38  # 会在运行时更新
    
    # 训练配置
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # 优化器配置
    OPTIMIZER = {
        'name': 'AdamW',
        'params': {
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'amsgrad': True
        }
    }
    
    # 学习率调度器配置
    SCHEDULER = {
        'name': 'CosineAnnealingWarmRestarts',
        'params': {
            'T_0': 10,
            'T_mult': 2,
            'eta_min': 1e-6
        }
    }
    
    # 早停配置
    EARLY_STOPPING_PATIENCE = 10  # 增加早停耐心值
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # GPU训练配置
    USE_AMP = True  # 使用混合精度训练
    GRAD_SCALER = True
    
    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取配置字典"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k.isupper()} 