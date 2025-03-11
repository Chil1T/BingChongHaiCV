from typing import Dict, Any
import torch
from .base_config import BaseConfig

class TrainConfig(BaseConfig):
    """训练配置类，继承自BaseConfig并添加特定的训练参数"""
    
    def __init__(self):
        super().__init__()
        
        # 数据集采样配置
        self.DATASET_SAMPLE_MODE = "percentage"  # 'percentage' 或 'fixed_size'
        self.DATASET_SAMPLE_SIZE = 1.0  # 百分比(0-1)或每类样本固定数量
        self.TRAIN_VAL_SPLIT = 0.2  # 验证集比例
        self.MIN_SAMPLES_PER_CLASS = 10  # 每类最小样本数
        self.BALANCED_SAMPLING = True  # 是否使用平衡采样
        
        # 数据增强配置
        self.AUGMENTATION = {
            'random_rotation': 20,  # 随机旋转角度范围
            'random_horizontal_flip': True,  # 随机水平翻转
            'random_vertical_flip': False,  # 随机垂直翻转
            'random_brightness': 0.2,  # 随机亮度调整范围
            'random_contrast': 0.2,  # 随机对比度调整范围
            'random_saturation': 0.2,  # 随机饱和度调整范围
            'random_hue': 0.1,  # 随机色调调整范围
            'normalize': True,  # 是否标准化
            'mean': [0.485, 0.456, 0.406],  # 标准化均值
            'std': [0.229, 0.224, 0.225]  # 标准化标准差
        }
        
        # 训练策略配置
        self.TRAINING = {
            'epochs': 100,  # 训练轮数
            'batch_size': 32,  # 批次大小
            'num_workers': 4,  # 数据加载线程数
            'pin_memory': True,  # 是否将数据加载到CUDA固定内存
            'grad_clip': 1.0,  # 梯度裁剪阈值
            'log_interval': 10,  # 日志打印间隔(批次)
            'eval_interval': 1,  # 评估间隔(轮次)
            'save_best_only': True,  # 是否只保存最佳模型
            'early_stopping_patience': 10,  # 早停耐心值
            'early_stopping_min_delta': 1e-4  # 早停最小增益
        }
        
        # 优化器配置
        self.OPTIMIZER = {
            'name': 'AdamW',  # 优化器名称
            'lr': 1e-3,  # 学习率
            'weight_decay': 1e-2,  # 权重衰减
            'beta1': 0.9,  # Adam beta1
            'beta2': 0.999  # Adam beta2
        }
        
        # 学习率调度器配置
        self.SCHEDULER = {
            'name': 'CosineAnnealingWarmRestarts',  # 调度器名称
            'T_0': 10,  # 初始周期
            'T_mult': 2,  # 周期倍增因子
            'eta_min': 1e-6  # 最小学习率
        }
    
    # 本地路径配置
    DATA_ROOT = "./data"
    MODEL_DIR = "./models"
    LOG_DIR = "./logs"
    
    # 覆盖基础配置
    TRAIN_DIR = f"{DATA_ROOT}/train"
    VAL_DIR = f"{DATA_ROOT}/val"
    BATCH_SIZE = 32  # 本地GPU内存可能较小，使用较小的batch size
    NUM_WORKERS = 4  # 本地多核CPU可以使用更多workers
    EPOCHS = 50  # 本地训练可以跑更多epoch
    
    # 检查点配置
    CHECKPOINT_DIR = MODEL_DIR
    BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pth"
    LAST_MODEL_PATH = f"{CHECKPOINT_DIR}/last_model.pth"
    
    # 数据集配置
    IMAGE_SIZE = 224
    PIN_MEMORY = True  # 启用内存钉扎加速数据传输
    
    # 模型配置
    MODEL_NAME = "resnet50"
    PRETRAINED = True
    NUM_CLASSES = 38  # 实际类别数
    
    # 训练配置
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    
    # 混合精度训练配置
    USE_AMP = True  # 启用自动混合精度
    GRAD_SCALER = True  # 使用梯度缩放器
    
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