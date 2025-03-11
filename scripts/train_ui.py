import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 确保在Windows上使用正确的事件循环策略
if os.name == 'nt':  # Windows系统
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        print(f"无法设置事件循环策略: {e}")

import streamlit as st
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import torch
import logging
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.config.train_config import TrainConfig
from model.core.model_factory import ModelFactory
from model.core.trainer import Trainer
from model.dataloader.dataset import create_dataloaders
from model.utils.dataset_sampler import DatasetSampler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingVisualizer:
    def __init__(self):
        self.config = TrainConfig()
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
    def update_plots(self):
        """更新训练过程图表"""
        if not self.history['train_loss']:
            return
            
        df = pd.DataFrame(self.history)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Loss', 'Accuracy'),
            vertical_spacing=0.15
        )
        
        # 添加损失曲线
        fig.add_trace(
            go.Scatter(y=df['train_loss'], name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=df['val_loss'], name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # 添加准确率曲线
        fig.add_trace(
            go.Scatter(y=df['train_acc'], name='Train Acc', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=df['val_acc'], name='Val Acc', line=dict(color='red')),
            row=2, col=1
        )
        
        # 更新布局
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text='Training Progress'
        )
        
        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
        
    def train_epoch_callback(self, epoch, train_metrics, val_metrics, lr):
        """每个epoch结束后的回调函数"""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['acc'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['acc'])
        self.history['lr'].append(lr)
        
        # 更新进度条
        progress_bar.progress((epoch + 1) / self.config.EPOCHS)
        
        # 更新指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train Loss", f"{train_metrics['loss']:.4f}")
        with col2:
            st.metric("Train Acc", f"{train_metrics['acc']:.2f}%")
        with col3:
            st.metric("Val Loss", f"{val_metrics['loss']:.4f}")
        with col4:
            st.metric("Val Acc", f"{val_metrics['acc']:.2f}%")
            
        # 更新图表
        self.update_plots()
        
    def prepare_dataset(self, source_dir, sample_size):
        """准备数据集"""
        self.config.DATASET_SAMPLE_SIZE = sample_size
        
        # 设置目标路径
        train_dir = os.path.join('data', 'train')
        val_dir = os.path.join('data', 'val')
        
        # 创建采样器并处理数据集
        sampler = DatasetSampler(self.config)
        sampler.process_dataset(source_dir, train_dir, val_dir)
        
    def train(self):
        """开始训练"""
        try:
            # 创建数据加载器
            train_loader, val_loader = create_dataloaders(self.config)
            
            # 直接创建模型
            model = ModelFactory.create_model(
                model_name=self.config.MODEL_NAME,
                num_classes=self.config.NUM_CLASSES,
                pretrained=self.config.PRETRAINED
            )
            
            # 直接创建优化器
            optimizer = ModelFactory.create_optimizer(
                model=model,
                optimizer_name='AdamW',
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                amsgrad=True
            )
            
            # 创建损失函数
            criterion = ModelFactory.create_criterion()
            
            # 创建学习率调度器
            scheduler = ModelFactory.create_scheduler(
                optimizer=optimizer,
                scheduler_name='CosineAnnealingWarmRestarts',
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
            
            # 创建训练器
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                config=self.config,
                device=self.config.DEVICE
            )
            
            # 设置进度条
            progress_bar = st.progress(0)
            
            # 开始训练
            for epoch in range(self.config.EPOCHS):
                # 训练一个epoch
                train_metrics = trainer.train_epoch(train_loader)
                
                # 验证
                val_metrics = trainer.validate(val_loader)
                
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                
                # 更新回调
                self.train_epoch_callback(
                    epoch,
                    train_metrics,
                    val_metrics,
                    current_lr
                )
                
                # 更新进度条
                progress_bar.progress((epoch + 1) / self.config.EPOCHS)
                
                # 学习率调度器步进
                if scheduler is not None:
                    scheduler.step()
                    
            # 保存最终模型
            os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
            trainer.save_checkpoint(self.config.BEST_MODEL_PATH)
            st.success(f"训练完成！模型保存在 {self.config.BEST_MODEL_PATH}")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            st.error(f"训练出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            st.stop()

def main():
    st.set_page_config(
        page_title="植物病虫害识别系统 - 训练面板",
        page_icon="🌿",
        layout="wide"
    )
    
    st.title("植物病虫害识别系统 - 训练面板")
    
    # 初始化训练器
    visualizer = TrainingVisualizer()
    
    # 侧边栏配置
    with st.sidebar:
        st.header("训练配置")
        
        # 数据集配置
        st.subheader("数据集配置")
        source_dir = st.text_input(
            "数据集目录",
            value="datasets/PlantVillage/color",
            help="原始数据集的目录路径"
        )
        
        sample_size = st.slider(
            "数据集采样比例",
            min_value=0.1,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="使用原始数据集的比例"
        )
        
        # 训练配置
        st.subheader("训练配置")
        visualizer.config.EPOCHS = st.number_input(
            "训练轮数",
            min_value=1,
            max_value=100,
            value=10
        )
        
        visualizer.config.BATCH_SIZE = st.number_input(
            "批次大小",
            min_value=1,
            max_value=128,
            value=32
        )
        
        visualizer.config.LEARNING_RATE = st.number_input(
            "学习率",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            format="%f"
        )
        
        # 开始训练按钮
        if st.button("开始训练"):
            try:
                # 准备数据集
                with st.spinner("准备数据集中..."):
                    visualizer.prepare_dataset(source_dir, sample_size)
                
                # 创建进度条
                global progress_bar
                progress_bar = st.progress(0)
                
                # 创建指标显示区
                st.subheader("训练指标")
                metrics_container = st.empty()
                
                # 开始训练
                start_time = datetime.now()
                visualizer.train()
                end_time = datetime.now()
                
                # 显示训练完成信息
                training_time = end_time - start_time
                st.success(f"训练完成！总用时: {training_time}")
                
            except Exception as e:
                st.error(f"训练过程中出现错误: {str(e)}")
                logger.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main() 