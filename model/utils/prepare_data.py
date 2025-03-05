import os
import shutil
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(source_dir, target_dir, test_size=0.2, random_state=42):
    """
    准备训练集和验证集
    
    Args:
        source_dir: PlantVillage数据集解压后的根目录
        target_dir: 处理后的数据集存放目录
        test_size: 验证集比例
        random_state: 随机种子
    """
    # 创建目标目录
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 获取所有类别
    categories = [d for d in os.listdir(source_dir) 
                 if os.path.isdir(os.path.join(source_dir, d))]
    
    for category in categories:
        logger.info(f"Processing category: {category}")
        
        # 创建目录
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        
        # 获取该类别下的所有图片
        category_dir = os.path.join(source_dir, category)
        images = [f for f in os.listdir(category_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 划分训练集和验证集
        train_images, val_images = train_test_split(
            images, test_size=test_size, random_state=random_state
        )
        
        # 复制训练集图片
        for img in tqdm(train_images, desc=f"Copying {category} training images"):
            src = os.path.join(category_dir, img)
            dst = os.path.join(train_dir, category, img)
            shutil.copy2(src, dst)
            
        # 复制验证集图片
        for img in tqdm(val_images, desc=f"Copying {category} validation images"):
            src = os.path.join(category_dir, img)
            dst = os.path.join(val_dir, category, img)
            shutil.copy2(src, dst)
    
    logger.info("Dataset preparation completed!")
    
    # 打印数据集统计信息
    print_dataset_stats(train_dir, val_dir)

def print_dataset_stats(train_dir, val_dir):
    """打印数据集统计信息"""
    print("\nDataset Statistics:")
    print("-" * 50)
    
    # 训练集统计
    train_categories = os.listdir(train_dir)
    print("\nTraining Set:")
    total_train = 0
    for category in train_categories:
        count = len(os.listdir(os.path.join(train_dir, category)))
        total_train += count
        print(f"{category}: {count} images")
    print(f"Total training images: {total_train}")
    
    # 验证集统计
    val_categories = os.listdir(val_dir)
    print("\nValidation Set:")
    total_val = 0
    for category in val_categories:
        count = len(os.listdir(os.path.join(val_dir, category)))
        total_val += count
        print(f"{category}: {count} images")
    print(f"Total validation images: {total_val}")

if __name__ == "__main__":
    # 使用示例
    source_dir = "path/to/plantvillage/dataset"  # PlantVillage数据集解压路径
    target_dir = "data"  # 处理后的数据集存放路径
    prepare_dataset(source_dir, target_dir) 