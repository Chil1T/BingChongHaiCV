import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_history():
    """加载训练历史数据"""
    history_path = project_root / "logs" / "training_history.json"
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        return history
    except FileNotFoundError:
        logger.error(f"Training history file not found: {history_path}")
        return None

def create_training_curves(history, output_dir):
    """创建训练曲线图"""
    if not history:
        return
    
    # 创建训练曲线
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Loss Curves', 'Accuracy Curves'),
        vertical_spacing=0.15
    )
    
    # 添加损失曲线
    fig.add_trace(
        go.Scatter(y=history['train_loss'], name='Train Loss', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history['val_loss'], name='Validation Loss', line=dict(color='#ff7f0e')),
        row=1, col=1
    )
    
    # 添加准确率曲线
    fig.add_trace(
        go.Scatter(y=history['train_acc'], name='Train Accuracy', line=dict(color='#2ca02c')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(y=history['val_acc'], name='Validation Accuracy', line=dict(color='#d62728')),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=800,
        title_text="Training Progress",
        showlegend=True,
        template='plotly_white'
    )
    
    # 保存图表
    fig.write_image(output_dir / "training_curves.png")
    fig.write_html(output_dir / "training_curves.html")
    logger.info("Training curves generated successfully")

def create_class_accuracy_chart(history, output_dir):
    """创建各类别准确率对比图"""
    if not history or 'class_accuracies' not in history:
        return
    
    # 准备数据
    classes = list(history['class_accuracies'].keys())
    accuracies = list(history['class_accuracies'].values())
    
    # 创建条形图
    fig = go.Figure(data=[
        go.Bar(
            x=accuracies,
            y=classes,
            orientation='h',
            marker=dict(color=accuracies, colorscale='Viridis')
        )
    ])
    
    # 更新布局
    fig.update_layout(
        title="Per-Class Accuracy",
        xaxis_title="Accuracy (%)",
        yaxis_title="Class",
        height=1200,
        template='plotly_white'
    )
    
    # 保存图表
    fig.write_image(output_dir / "class_accuracies.png")
    fig.write_html(output_dir / "class_accuracies.html")
    logger.info("Class accuracy chart generated successfully")

def create_confusion_matrix(history, output_dir):
    """创建混淆矩阵热力图"""
    if not history or 'confusion_matrix' not in history:
        return
    
    # 准备数据
    cm = np.array(history['confusion_matrix'])
    classes = history.get('class_names', [f"Class {i}" for i in range(cm.shape[0])])
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale='Viridis',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    # 更新布局
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        height=1000,
        width=1000,
        template='plotly_white'
    )
    
    # 保存图表
    fig.write_image(output_dir / "confusion_matrix.png")
    fig.write_html(output_dir / "confusion_matrix.html")
    logger.info("Confusion matrix generated successfully")

def main():
    # 创建输出目录
    output_dir = project_root / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载训练历史
    history = load_training_history()
    if not history:
        logger.error("Failed to load training history")
        return
    
    # 生成可视化图表
    create_training_curves(history, output_dir)
    create_class_accuracy_chart(history, output_dir)
    create_confusion_matrix(history, output_dir)
    
    logger.info(f"All visualizations have been saved to {output_dir}")

if __name__ == "__main__":
    main() 