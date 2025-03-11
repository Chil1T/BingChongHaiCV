import torch
from typing import Dict, Any

def get_device() -> str:
    """获取可用的计算设备
    
    Returns:
        str: 设备名称 ('cuda', 'rocm', 'mps', 或 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'hip') and torch.hip.is_available():  # 检查ROCm支持
        return "cuda"  # ROCm也使用cuda作为设备名
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # 检查MPS支持
        return "mps"
    return "cpu"

class BaseConfig:
    """基础配置类"""
    
    # 数据集配置
    DATA_DIR = "data"
    TRAIN_DIR = f"{DATA_DIR}/train"
    VAL_DIR = f"{DATA_DIR}/val"
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 6
    PIN_MEMORY = True
    
    # 数据集采样和分割配置
    DATASET_SAMPLE_MODE = "percentage"  # 'percentage' 或 'fixed_size'
    DATASET_SAMPLE_SIZE = 1.0  # 如果是百分比模式，范围0-1；如果是固定大小模式，表示每个类别的样本数
    TRAIN_VAL_SPLIT = 0.2  # 验证集比例
    RANDOM_SEED = 42  # 随机种子，确保可复现性
    MIN_SAMPLES_PER_CLASS = 10  # 每个类别的最小样本数
    BALANCED_SAMPLING = True  # 是否使用平衡采样
    
    # 模型配置
    MODEL_NAME = "resnet50"
    PRETRAINED = True
    NUM_CLASSES = 38  # 会在运行时更新
    
    # 训练配置
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # 优化器配置
    OPTIMIZER = {
        'name': 'AdamW',
        'params': {
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
    EARLY_STOPPING_PATIENCE = 7
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # 混合精度训练配置
    USE_AMP = True
    GRAD_SCALER = True
    
    # 设备配置
    DEVICE = get_device()
    
    # 检查点配置
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pth"
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取配置字典"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k.isupper()} 