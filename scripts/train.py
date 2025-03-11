import os
import logging
from model.config.train_config import TrainConfig
from model.core.model_factory import ModelFactory
from model.core.trainer import Trainer
from model.dataloader.dataset import create_dataloaders

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    # 创建配置对象
    config = TrainConfig()
    
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
    
if __name__ == '__main__':
    main() 