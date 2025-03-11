# 植物病虫害识别模型验证

本分支专门用于模型验证，包含多个验证数据集的生成和评估脚本。

## 目录结构

```
evaluation/
├── notebooks/          # Colab notebooks
│   └── evaluate_model.ipynb
├── scripts/           # 验证相关脚本
│   ├── prepare_validation_sets.py
│   └── evaluate_model.py
└── requirements.txt   # 依赖包
```

## 使用方法

1. 在Google Drive中创建项目目录：
```
plant_disease_project/
├── data/
│   ├── train/        # 训练数据（可选）
│   └── val/          # 验证数据
├── models/
│   └── best_model.pth  # 训练好的模型
└── results/          # 评估结果将保存在这里
```

2. 在Colab中运行：

```python
# 克隆项目代码（evaluation分支）
!git clone -b evaluation https://github.com/your-username/BingChongHaiCV.git
%cd BingChongHaiCV

# 安装依赖
!pip install -r requirements.txt

# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 创建必要的目录
!mkdir -p data/val data/val_variations models logs
```

3. 准备验证数据集：
   - 将原始验证数据放在 `data/val` 目录
   - 运行 `prepare_validation_sets.py` 生成其他验证集变体

4. 运行评估：
   - 使用 `evaluate_model.py` 在所有验证集上评估模型
   - 结果将保存在 `logs/evaluation_results.json`

## 验证数据集说明

1. **原始验证集**：
   - 来源：PlantVillage数据集的验证部分
   - 用途：基准性能评估

2. **变体验证集**：
   - 光照变化：测试模型在不同光照条件下的表现
     - 弱光版本
     - 强光版本
   - 噪声测试：评估模型的抗干扰能力
     - 高斯噪声
     - 椒盐噪声
   - 遮挡测试：验证模型对部分特征的识别能力
     - 随机遮挡

## 评估指标

1. 总体准确率
2. 每个类别的准确率
3. 混淆矩阵
4. 各验证集的对比分析

## 注意事项

1. 确保GPU可用以加速评估过程
2. 评估大约需要1-2小时（取决于数据集大小）
3. 建议先用小数据集测试流程
4. 所有结果会自动保存到Google Drive 