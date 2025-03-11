import os
import sys
import logging
from pathlib import Path
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.utils.dataset_sampler import DatasetSampler
from model.config.train_config import TrainConfig

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='准备训练数据集')
    
    parser.add_argument('--source-dir', type=str, required=True,
                      help='源数据集目录路径')
    parser.add_argument('--target-dir', type=str, default='data',
                      help='处理后的数据集存放路径')
    parser.add_argument('--sample-mode', type=str, choices=['percentage', 'fixed_size'],
                      default='percentage',
                      help='采样模式：按百分比或固定数量')
    parser.add_argument('--sample-size', type=float, default=1.0,
                      help='采样大小：百分比(0-1)或每类样本数量')
    parser.add_argument('--val-split', type=float, default=0.2,
                      help='验证集比例(0-1)')
    parser.add_argument('--min-samples', type=int, default=10,
                      help='每个类别的最小样本数')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    parser.add_argument('--balanced', action='store_true',
                      help='是否使用平衡采样')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 确保使用绝对路径
        source_dir = os.path.abspath(args.source_dir)
        target_dir = os.path.abspath(args.target_dir)
        
        logger.info(f"使用源目录: {source_dir}")
        logger.info(f"使用目标目录: {target_dir}")
        
        # 检查源目录是否存在
        if not os.path.exists(source_dir):
            logger.error(f"Source directory not found: {source_dir}")
            logger.error("Please make sure you have placed the dataset in the correct location.")
            sys.exit(1)
            
        # 创建配置对象
        config = TrainConfig()
        
        # 更新配置
        config.DATASET_SAMPLE_MODE = args.sample_mode
        config.DATASET_SAMPLE_SIZE = args.sample_size
        config.TRAIN_VAL_SPLIT = args.val_split
        config.MIN_SAMPLES_PER_CLASS = args.min_samples
        config.RANDOM_SEED = args.seed
        config.BALANCED_SAMPLING = args.balanced
        
        # 设置目标路径
        train_dir = os.path.join(target_dir, 'train')
        val_dir = os.path.join(target_dir, 'val')
        
        # 创建采样器并处理数据集
        logger.info("Starting dataset preparation...")
        sampler = DatasetSampler(config)
        sampler.process_dataset(source_dir, train_dir, val_dir)
        
        logger.info("Dataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during dataset preparation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 