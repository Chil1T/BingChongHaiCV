import os
import sys
from pathlib import Path
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.dataloader.dataset import create_dataloaders
from model.core.model_factory import ModelFactory
from model.config.train_config import TrainConfig

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, val_loader, device):
    """在验证集上评估模型，收集详细的性能指标"""
    model.eval()
    all_preds = []
    all_labels = []
    class_correct = {}
    class_total = {}
    
    # 初始化每个类别的统计
    for class_name in val_loader.dataset.classes:
        class_correct[class_name] = 0
        class_total[class_name] = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # 收集预测结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 统计每个类别的正确数和总数
            for label, pred in zip(labels, preds):
                class_name = val_loader.dataset.classes[label]
                class_total[class_name] += 1
                if label == pred:
                    class_correct[class_name] += 1
    
    # 计算每个类别的准确率
    class_accuracies = {
        class_name: (class_correct[class_name] / class_total[class_name] * 100)
        for class_name in class_correct.keys()
    }
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm.tolist(),
        'class_names': val_loader.dataset.classes
    }

def main():
    try:
        # 创建配置对象
        config = TrainConfig()
        
        # 创建数据加载器
        _, val_loader = create_dataloaders(config)
        logger.info("Data loaders created")
        
        # 创建模型
        model = ModelFactory.create_model(
            model_name=config.MODEL_NAME,
            num_classes=len(val_loader.dataset.classes),
            pretrained=False
        )
        
        # 加载最佳模型权重
        model_path = os.path.join(project_root, 'models', 'best_model.pth')
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return
            
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded successfully")
        
        # 将模型移到适当的设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 评估模型
        logger.info("Starting model evaluation...")
        eval_results = evaluate_model(model, val_loader, device)
        
        # 加载现有的训练历史
        history_path = os.path.join(project_root, 'logs', 'training_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # 保留原有的训练曲线数据
        train_history = {
            'train_loss': history.get('train_loss', []),
            'val_loss': history.get('val_loss', []),
            'train_acc': history.get('train_acc', []),
            'val_acc': history.get('val_acc', [])
        }
        
        # 更新训练历史，同时保留训练曲线数据
        history = {**train_history, **eval_results}
        
        # 保存更新后的训练历史
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        logger.info("Evaluation completed and results saved")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 