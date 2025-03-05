import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.utils.prepare_data import prepare_dataset

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 设置路径
    source_dir = os.path.join(project_root, "datasets", "PlantVillage", "color")  # 使用彩色图像目录
    target_dir = os.path.join(project_root, "data")
    
    # 数据集划分参数
    test_size = 0.2  # 验证集比例
    random_state = 42  # 随机种子
    
    try:
        # 检查源目录是否存在
        if not os.path.exists(source_dir):
            logger.error(f"Source directory not found: {source_dir}")
            logger.error("Please make sure you have placed the PlantVillage dataset in the correct location.")
            sys.exit(1)
            
        # 检查数据集结构
        categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        logger.info(f"Found {len(categories)} categories in the dataset:")
        for category in categories:
            category_path = os.path.join(source_dir, category)
            num_images = len([f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            logger.info(f"  - {category}: {num_images} images")
            
        # 运行数据预处理
        logger.info("\nStarting dataset preparation...")
        logger.info(f"Source directory: {source_dir}")
        logger.info(f"Target directory: {target_dir}")
        
        prepare_dataset(
            source_dir=source_dir,
            target_dir=target_dir,
            test_size=test_size,
            random_state=random_state
        )
        logger.info("Dataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during dataset preparation: {str(e)}")
        sys.exit(1) 