from typing import Dict, Any
import torch

class ColabTrainingConfig:
    # 路径配置
    DRIVE_ROOT = "/content/drive/MyDrive/plant_disease_project"
    DATA_ROOT = f"{DRIVE_ROOT}/data"
    MODEL_DIR = f"{DRIVE_ROOT}/models"
    LOG_DIR = f"{DRIVE_ROOT}/logs"
    
    # 数据集配置
    TRAIN_DIR = f"{DATA_ROOT}/train"
    VAL_DIR = f"{DATA_ROOT}/val"
    IMAGE_SIZE = 224
    BATCH_SIZE = 64  # GPU版本可以用更大的batch size
    NUM_WORKERS = 2  # Colab环境建议使用2-4
    PIN_MEMORY = True
    
    # 模型配置
    MODEL_NAME = "resnet50"
    PRETRAINED = True
    NUM_CLASSES = 38  # 会在运行时更新
    
    # 训练配置
    EPOCHS = 10  # 小数据集先跑10个epoch看效果
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
    EARLY_STOPPING_PATIENCE = 7
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # GPU训练配置
    USE_AMP = True  # 使用混合精度训练
    GRAD_SCALER = True
    
    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 检查点配置
    CHECKPOINT_DIR = MODEL_DIR
    BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pth"
    LAST_MODEL_PATH = f"{CHECKPOINT_DIR}/last_model.pth"
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取配置字典"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k.isupper()} 