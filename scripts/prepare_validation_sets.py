import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationSetGenerator:
    """验证数据集生成器"""
    
    def __init__(self, source_dir: str, output_base_dir: str):
        self.source_dir = Path(source_dir)
        self.output_base_dir = Path(output_base_dir)
        
    def create_lighting_variations(self, image):
        """创建不同光照条件的变体"""
        # 弱光版本
        low_light = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.3), contrast_limit=(0, 0), p=1),
        ])(image=image)['image']
        
        # 强光版本
        high_light = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(0.3, 0.3), contrast_limit=(0, 0), p=1),
        ])(image=image)['image']
        
        return {'low_light': low_light, 'high_light': high_light}
    
    def create_noise_variations(self, image):
        """创建不同噪声的变体"""
        # 高斯噪声
        gaussian_noise = A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
        ])(image=image)['image']
        
        # 椒盐噪声
        sp_noise = image.copy()
        prob = 0.02
        black = np.random.random(image.shape[:2])
        white = np.random.random(image.shape[:2])
        sp_noise[black < prob/2] = 0
        sp_noise[white > 1 - prob/2] = 255
        
        return {'gaussian_noise': gaussian_noise, 'sp_noise': sp_noise}
    
    def create_occlusion_variations(self, image):
        """创建遮挡变体"""
        # 随机遮挡
        random_occluded = A.Compose([
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1),
        ])(image=image)['image']
        
        return {'random_occlusion': random_occluded}
    
    def process_image(self, src_path: Path, variation_type: str):
        """处理单张图片"""
        try:
            # 读取图片
            image = cv2.imread(str(src_path))
            if image is None:
                logger.error(f"Failed to read image: {src_path}")
                return
            
            # 创建变体
            if variation_type == 'lighting':
                variations = self.create_lighting_variations(image)
            elif variation_type == 'noise':
                variations = self.create_noise_variations(image)
            elif variation_type == 'occlusion':
                variations = self.create_occlusion_variations(image)
            else:
                logger.error(f"Unknown variation type: {variation_type}")
                return
            
            # 保存变体
            for var_name, var_image in variations.items():
                # 构建输出路径
                rel_path = src_path.relative_to(self.source_dir)
                out_dir = self.output_base_dir / var_name / rel_path.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存图片
                cv2.imwrite(str(out_dir / rel_path.name), var_image)
                
        except Exception as e:
            logger.error(f"Error processing {src_path}: {str(e)}")
    
    def create_validation_sets(self):
        """创建所有验证集"""
        try:
            # 获取所有图片文件
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(self.source_dir.rglob(f"*{ext}"))
            
            # 创建变体
            variation_types = ['lighting', 'noise', 'occlusion']
            
            for variation_type in variation_types:
                logger.info(f"Creating {variation_type} variations...")
                with ThreadPoolExecutor() as executor:
                    process_func = partial(self.process_image, variation_type=variation_type)
                    list(tqdm(executor.map(process_func, image_files), 
                            total=len(image_files),
                            desc=f"Processing {variation_type}"))
            
            logger.info("All validation sets created successfully!")
            
        except Exception as e:
            logger.error(f"Error creating validation sets: {str(e)}")
            raise

def main():
    # 设置路径
    project_root = Path(__file__).resolve().parent.parent
    source_dir = project_root / "data" / "val"
    output_base_dir = project_root / "data" / "val_variations"
    
    # 创建验证集生成器
    generator = ValidationSetGenerator(source_dir, output_base_dir)
    
    # 生成验证集
    generator.create_validation_sets()
    
    # 打印统计信息
    print("\nValidation Sets Statistics:")
    print("-" * 50)
    for var_dir in output_base_dir.iterdir():
        if var_dir.is_dir():
            image_count = sum(1 for _ in var_dir.rglob("*.jpg"))
            print(f"{var_dir.name}: {image_count} images")

if __name__ == "__main__":
    main() 