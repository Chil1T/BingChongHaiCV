import os
import cv2
import numpy as np
from tqdm import tqdm
import random
import shutil
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationSetGenerator:
    def __init__(self, source_dir, output_base_dir):
        """
        初始化验证集生成器
        
        Args:
            source_dir (str): 原始验证数据集目录
            output_base_dir (str): 输出目录基础路径
        """
        self.source_dir = Path(source_dir)
        self.output_base_dir = Path(output_base_dir)
        self.variations = {
            'lighting': ['dark', 'bright'],
            'noise': ['gaussian', 'salt_pepper'],
            'occlusion': ['random']
        }
        
    def setup_directories(self):
        """创建必要的输出目录"""
        for variation_type in self.variations:
            for variant in self.variations[variation_type]:
                output_dir = self.output_base_dir / f"{variation_type}_{variant}"
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {output_dir}")
                
    def apply_lighting_variation(self, img, variant):
        """
        应用光照变化
        
        Args:
            img (np.ndarray): 输入图像
            variant (str): 变化类型 ('dark' 或 'bright')
            
        Returns:
            np.ndarray: 处理后的图像
        """
        if variant == 'dark':
            return cv2.convertScaleAbs(img, alpha=0.6, beta=-30)
        elif variant == 'bright':
            return cv2.convertScaleAbs(img, alpha=1.4, beta=30)
        return img
    
    def apply_noise(self, img, variant):
        """
        添加噪声
        
        Args:
            img (np.ndarray): 输入图像
            variant (str): 噪声类型 ('gaussian' 或 'salt_pepper')
            
        Returns:
            np.ndarray: 添加噪声后的图像
        """
        if variant == 'gaussian':
            row, col, ch = img.shape
            mean = 0
            sigma = 25
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = img + gauss
            return np.clip(noisy, 0, 255).astype(np.uint8)
        elif variant == 'salt_pepper':
            s_vs_p = 0.5
            amount = 0.004
            noisy = np.copy(img)
            # Salt
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                     for i in img.shape]
            noisy[tuple(coords)] = 255
            # Pepper
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                     for i in img.shape]
            noisy[tuple(coords)] = 0
            return noisy
        return img
    
    def apply_occlusion(self, img, variant):
        """
        添加遮挡
        
        Args:
            img (np.ndarray): 输入图像
            variant (str): 遮挡类型 ('random')
            
        Returns:
            np.ndarray: 添加遮挡后的图像
        """
        if variant == 'random':
            h, w = img.shape[:2]
            # 随机生成遮挡区域（图像面积的10-20%）
            occlude_area = random.uniform(0.1, 0.2)
            occlude_size = int(np.sqrt(h * w * occlude_area))
            x = random.randint(0, w - occlude_size)
            y = random.randint(0, h - occlude_size)
            
            img_with_occlusion = img.copy()
            img_with_occlusion[y:y+occlude_size, x:x+occlude_size] = 0
            return img_with_occlusion
        return img
    
    def process_image(self, img_path, variation_type, variant):
        """
        处理单张图像
        
        Args:
            img_path (Path): 输入图像路径
            variation_type (str): 变化类型
            variant (str): 具体变体
            
        Returns:
            np.ndarray: 处理后的图像
        """
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Failed to read image: {img_path}")
            return None
            
        if variation_type == 'lighting':
            return self.apply_lighting_variation(img, variant)
        elif variation_type == 'noise':
            return self.apply_noise(img, variant)
        elif variation_type == 'occlusion':
            return self.apply_occlusion(img, variant)
        return img
    
    def generate_variations(self):
        """生成所有验证集变体"""
        self.setup_directories()
        
        # 遍历所有类别
        for class_dir in tqdm(list(self.source_dir.glob('*')), desc="Processing classes"):
            if not class_dir.is_dir():
                continue
                
            # 对每种变化类型
            for variation_type in self.variations:
                for variant in self.variations[variation_type]:
                    # 创建对应的输出类别目录
                    output_class_dir = self.output_base_dir / f"{variation_type}_{variant}" / class_dir.name
                    output_class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 处理该类别下的所有图像
                    for img_path in class_dir.glob('*.*'):
                        if not img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            continue
                            
                        processed_img = self.process_image(img_path, variation_type, variant)
                        if processed_img is not None:
                            output_path = output_class_dir / img_path.name
                            cv2.imwrite(str(output_path), processed_img)
                            
        logger.info("Validation set generation completed!")

def main():
    # 配置参数
    source_dir = "data/val"  # 原始验证集目录
    output_base_dir = "data/val_variations"  # 变体输出目录
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        logger.error(f"Source directory {source_dir} does not exist!")
        return
        
    # 创建验证集生成器并执行
    generator = ValidationSetGenerator(source_dir, output_base_dir)
    generator.generate_variations()
    
if __name__ == "__main__":
    main() 