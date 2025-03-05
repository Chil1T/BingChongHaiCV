from typing import Dict, Any
import torch

class TrainingConfig:
    # 数据集配置
    DATA_DIR = "data"
    TRAIN_DIR = f"{DATA_DIR}/train"
    VAL_DIR = f"{DATA_DIR}/val"
    IMAGE_SIZE = 224
    BATCH_SIZE = 32  # 根据GPU显存调整
    NUM_WORKERS = 6  # 设置为CPU核心数
    PIN_MEMORY = True  # 启用内存钉扎加速数据传输
    
    # 模型配置
    MODEL_NAME = "resnet50"
    PRETRAINED = True
    NUM_CLASSES = 38  # 实际类别数
    
    # 训练配置
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    
    # 优化器配置
    OPTIMIZER = {
        'name': 'AdamW',  # 使用AdamW优化器
        'params': {
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'amsgrad': True  # 启用AMSGrad
        }
    }
    
    # 学习率调度器配置
    SCHEDULER = {
        'name': 'CosineAnnealingWarmRestarts',
        'params': {
            'T_0': 10,  # 初始周期
            'T_mult': 2,  # 周期倍增
            'eta_min': 1e-6  # 最小学习率
        }
    }
    
    # 早停配置
    EARLY_STOPPING_PATIENCE = 7
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # 混合精度训练配置
    USE_AMP = True  # 启用自动混合精度
    GRAD_SCALER = True  # 使用梯度缩放器
    
    # 保存配置
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pth"
    
    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        # 检查是否支持MPS（Metal Performance Shaders，用于Mac）
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = "mps"
        # 检查是否支持ROCm（AMD GPU）
        elif hasattr(torch, "hip") and torch.hip.is_available():
            DEVICE = "cuda"  # ROCm使用cuda作为设备名
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取配置字典"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k.isupper()} 