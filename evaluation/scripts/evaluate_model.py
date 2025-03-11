import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from sklearn.metrics import confusion_matrix, classification_report
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path, device=None):
        """
        初始化模型评估器
        
        Args:
            model_path (str): 模型文件路径
            device (str, optional): 运行设备
        """
        self.model_path = Path(model_path)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self):
        """加载模型"""
        try:
            # 加载模型
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def create_dataloader(self, data_dir):
        """
        创建数据加载器
        
        Args:
            data_dir (str): 数据目录
            
        Returns:
            DataLoader: 数据加载器
            list: 类别列表
        """
        try:
            dataset = datasets.ImageFolder(data_dir, transform=self.transform)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            return dataloader, dataset.classes
        except Exception as e:
            logger.error(f"Failed to create dataloader for {data_dir}: {e}")
            raise
            
    def evaluate(self, dataloader):
        """
        评估模型
        
        Args:
            dataloader (DataLoader): 数据加载器
            
        Returns:
            tuple: (预测标签列表, 真实标签列表, 每个样本的预测概率)
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
        
    def calculate_metrics(self, preds, labels, probs, classes):
        """
        计算评估指标
        
        Args:
            preds (np.ndarray): 预测标签
            labels (np.ndarray): 真实标签
            probs (np.ndarray): 预测概率
            classes (list): 类别列表
            
        Returns:
            dict: 评估指标
        """
        # 计算混淆矩阵
        cm = confusion_matrix(labels, preds)
        
        # 计算每个类别的准确率
        class_report = classification_report(labels, preds, 
                                          target_names=classes, 
                                          output_dict=True)
        
        # 计算总体准确率
        accuracy = (preds == labels).mean()
        
        # 计算每个类别的准确率和置信度
        class_metrics = {}
        for i, class_name in enumerate(classes):
            class_mask = (labels == i)
            if np.any(class_mask):
                class_acc = (preds[class_mask] == labels[class_mask]).mean()
                class_conf = probs[class_mask, i].mean()
                class_metrics[class_name] = {
                    'accuracy': float(class_acc),
                    'confidence': float(class_conf)
                }
        
        return {
            'overall_accuracy': float(accuracy),
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
        
def main():
    # 配置参数
    model_path = "models/best_model.pth"
    validation_dirs = {
        'original': 'data/val',
        'lighting_dark': 'data/val_variations/lighting_dark',
        'lighting_bright': 'data/val_variations/lighting_bright',
        'noise_gaussian': 'data/val_variations/noise_gaussian',
        'noise_salt_pepper': 'data/val_variations/noise_salt_pepper',
        'occlusion_random': 'data/val_variations/occlusion_random'
    }
    results_file = "logs/evaluation_results.json"
    
    # 创建评估器
    evaluator = ModelEvaluator(model_path)
    evaluator.load_model()
    
    # 存储所有结果
    all_results = {}
    
    # 评估每个验证集
    for val_name, val_dir in validation_dirs.items():
        if not os.path.exists(val_dir):
            logger.warning(f"Validation directory {val_dir} does not exist, skipping...")
            continue
            
        logger.info(f"\nEvaluating {val_name} validation set...")
        
        # 创建数据加载器
        dataloader, classes = evaluator.create_dataloader(val_dir)
        
        # 评估模型
        start_time = time.time()
        preds, labels, probs = evaluator.evaluate(dataloader)
        eval_time = time.time() - start_time
        
        # 计算指标
        metrics = evaluator.calculate_metrics(preds, labels, probs, classes)
        metrics['evaluation_time'] = eval_time
        
        # 存储结果
        all_results[val_name] = metrics
        
        # 输出主要结果
        logger.info(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
        logger.info(f"Evaluation time: {eval_time:.2f} seconds")
        
    # 保存结果
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"\nResults saved to {results_file}")
    
if __name__ == "__main__":
    main() 