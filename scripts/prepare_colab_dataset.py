import os
import sys
import random
import shutil
from pathlib import Path
import logging
from tqdm import tqdm
import zipfile

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_small_dataset(source_dir, target_dir, samples_per_class=20, seed=42):
    """
    从源目录创建小规模数据集
    
    Args:
        source_dir: 源数据集目录
        target_dir: 目标目录
        samples_per_class: 每个类别选择的样本数
        seed: 随机种子
    """
    random.seed(seed)
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有类别
    categories = [d for d in os.listdir(source_dir) 
                 if os.path.isdir(os.path.join(source_dir, d))]
    
    total_copied = 0
    for category in tqdm(categories, desc="Processing categories"):
        source_category_dir = os.path.join(source_dir, category)
        target_category_dir = os.path.join(target_dir, category)
        os.makedirs(target_category_dir, exist_ok=True)
        
        # 获取该类别下的所有图片
        images = [f for f in os.listdir(source_category_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 随机选择指定数量的图片
        selected_images = random.sample(
            images, 
            min(samples_per_class, len(images))
        )
        
        # 复制选中的图片
        for img in selected_images:
            src = os.path.join(source_category_dir, img)
            dst = os.path.join(target_category_dir, img)
            shutil.copy2(src, dst)
            total_copied += 1
            
    return total_copied

def create_zip_archive(source_dir, zip_path):
    """
    创建数据集的zip压缩文件
    
    Args:
        source_dir: 要压缩的目录
        zip_path: 压缩文件保存路径
    """
    logger.info(f"Creating zip archive: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in tqdm(files, desc="Compressing files"):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)

def main():
    # 设置路径
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    colab_data_dir = project_root / "colab_data"
    
    # 创建小规模数据集目录
    colab_train_dir = colab_data_dir / "train"
    colab_val_dir = colab_data_dir / "val"
    
    try:
        # 处理训练集
        logger.info("Processing training set...")
        train_copied = create_small_dataset(
            source_dir=data_dir / "train",
            target_dir=colab_train_dir,
            samples_per_class=20
        )
        logger.info(f"Copied {train_copied} training images")
        
        # 处理验证集
        logger.info("Processing validation set...")
        val_copied = create_small_dataset(
            source_dir=data_dir / "val",
            target_dir=colab_val_dir,
            samples_per_class=5
        )
        logger.info(f"Copied {val_copied} validation images")
        
        # 创建zip文件
        zip_path = project_root / "colab_data.zip"
        create_zip_archive(colab_data_dir, zip_path)
        
        # 计算压缩文件大小
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        logger.info(f"Created zip archive: {zip_path} ({zip_size_mb:.2f}MB)")
        
        # 清理临时目录
        shutil.rmtree(colab_data_dir)
        logger.info("Cleaned up temporary directory")
        
        logger.info("""
数据集准备完成！

后续步骤：
1. 在您的Google Drive中创建 'plant_disease_project' 文件夹
2. 在其中创建 'data' 子文件夹
3. 将生成的 'colab_data.zip' 上传到 'plant_disease_project' 文件夹
4. 在Colab中解压数据集

生成的数据集信息：
- 训练集图片数量: {train_copied}
- 验证集图片数量: {val_copied}
- 压缩文件大小: {zip_size_mb:.2f}MB
""")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        if os.path.exists(colab_data_dir):
            shutil.rmtree(colab_data_dir)
        sys.exit(1)

if __name__ == "__main__":
    main() 