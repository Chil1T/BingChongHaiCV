import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import logging
from model.config.colab_config import ColabConfig as cfg
from model.core.model_factory import ModelFactory
from model.core.trainer import Trainer
from model.dataloader.dataset import create_dataloaders

def setup_directories():
    """创建必要的目录"""
    os.makedirs(cfg.DATA_ROOT, exist_ok=True)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

# 首先创建必要的目录
setup_directories()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(cfg.LOG_DIR, 'training.log'))
    ]
)
logger = logging.getLogger(__name__)

def setup_colab_env():
    """设置 Colab 环境"""
    logger.info("Colab environment setup completed")

def main():
    """主函数"""
    try:
        # 设置环境
        setup_colab_env()
        
        # 创建配置对象
        config = cfg()
        
        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(config)
        logger.info("Data loaders created")
        
        # 创建训练组件
        model, optimizer, criterion, scheduler = ModelFactory.create_training_components(config)
        logger.info("Training components created")
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            config=config,
            device=config.DEVICE
        )
        logger.info("Trainer created")
        
        # 加载检查点（如果存在）
        if os.path.exists(config.LAST_MODEL_PATH):
            trainer.load_checkpoint(config.LAST_MODEL_PATH)
            logger.info("Checkpoint loaded")
        
        # 开始训练
        logger.info("Starting training...")
        history = trainer.train(train_loader, val_loader)
        logger.info("Training completed")
        
        return history
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 