import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_plot_style():
    """设置图表样式"""
    plt.style.use('seaborn')
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = [10, 6]

def create_training_curves(history, save_dir):
    """创建训练曲线图（损失和准确率）"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 绘制损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失')
    ax1.set_title('训练过程中的损失变化')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率')
    ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率')
    ax2.set_title('训练过程中的准确率变化')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_class_accuracy_chart(history, save_dir):
    """创建各类别准确率对比图"""
    classes = list(history['class_accuracies'].keys())
    accuracies = list(history['class_accuracies'].values())
    
    # 创建水平条形图
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(classes))
    
    # 绘制条形图
    bars = plt.barh(y_pos, accuracies)
    plt.yticks(y_pos, [c.replace('___', '\n') for c in classes])
    plt.xlabel('准确率 (%)')
    plt.title('各类别识别准确率')
    
    # 在条形上添加数值标签
    for i, v in enumerate(accuracies):
        plt.text(v, i, f' {v:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_accuracies.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix(history, save_dir):
    """创建混淆矩阵热力图"""
    confusion_mat = np.array(history['confusion_matrix'])
    class_names = history['class_names']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置路径
    project_root = Path(__file__).resolve().parent.parent
    history_file = project_root / 'logs' / 'training_history.json'
    figures_dir = project_root / 'docs' / 'figures'
    
    # 创建保存图表的目录
    os.makedirs(figures_dir, exist_ok=True)
    
    # 设置图表样式
    setup_plot_style()
    
    # 读取训练历史数据
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # 生成图表
    print("正在生成训练曲线图...")
    create_training_curves(history, figures_dir)
    
    print("正在生成类别准确率对比图...")
    create_class_accuracy_chart(history, figures_dir)
    
    print("正在生成混淆矩阵图...")
    create_confusion_matrix(history, figures_dir)
    
    print(f"\n所有图表已生成完成，保存在: {figures_dir}")

if __name__ == '__main__':
    main() 