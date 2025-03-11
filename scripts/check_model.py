#!/usr/bin/env python
"""
检查模型文件的结构
"""

import os
import sys
import torch
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 模型文件路径
    model_path = os.path.join(root_dir, 'models', 'best_model.pth')
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    logger.info(f"加载模型文件: {model_path}")
    
    # 加载模型文件
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 打印模型文件的结构
        logger.info("模型文件结构:")
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                if isinstance(value, dict):
                    logger.info(f"- {key}: 字典，包含 {len(value)} 个键")
                elif isinstance(value, torch.Tensor):
                    logger.info(f"- {key}: 张量，形状 {value.shape}")
                else:
                    logger.info(f"- {key}: {type(value)}")
            
            # 如果包含model_state_dict，检查其结构
            if "model_state_dict" in checkpoint:
                logger.info("\nmodel_state_dict结构:")
                state_dict = checkpoint["model_state_dict"]
                # 打印前10个键
                for i, (key, value) in enumerate(state_dict.items()):
                    if i < 10:
                        logger.info(f"- {key}: 张量，形状 {value.shape}")
                    else:
                        break
                logger.info(f"... 共 {len(state_dict)} 个键")
        else:
            logger.info(f"模型文件不是字典，而是 {type(checkpoint)}")
    
    except Exception as e:
        logger.error(f"加载模型文件时出错: {str(e)}")

if __name__ == "__main__":
    main() 