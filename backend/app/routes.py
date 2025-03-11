from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time
import logging
import random
from logging.handlers import RotatingFileHandler
from backend.app.utils.file_util import allowed_file, get_model_path, get_class_names

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        RotatingFileHandler(
            'app.log',
            maxBytes=100*1024*1024,  # 100MB
            backupCount=5
        )
    ]
)
logger = logging.getLogger(__name__)

api = Blueprint('api', __name__)

# 全局变量
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = get_class_names()
DEMO_MODE = False  # 演示模式标志

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 加载模型
def load_model():
    """加载模型，如果模型已加载则返回已加载的模型"""
    global model, DEMO_MODE
    
    if model is not None:
        return model
    
    try:
        model_path = get_model_path()
        
        # 如果没有找到模型文件，进入演示模式
        if model_path is None:
            logger.warning("未找到模型文件，进入演示模式")
            DEMO_MODE = True
            return None
        
        logger.info(f"加载模型: {model_path}")
        
        # 创建模型架构
        model = models.resnet50(pretrained=False)
        num_classes = len(CLASSES)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"加载的checkpoint类型: {type(checkpoint)}")
        
        # 根据检查结果，我们知道checkpoint是一个字典，包含model_state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            logger.info("从checkpoint中提取model_state_dict")
            state_dict = checkpoint["model_state_dict"]
            model.load_state_dict(state_dict)
            logger.info("成功加载model_state_dict")
        else:
            logger.error("checkpoint中没有model_state_dict")
            logger.warning("进入演示模式")
            DEMO_MODE = True
            return None
        
        model.to(device)
        model.eval()
        
        logger.info(f"模型加载成功，类别数: {num_classes}")
        return model
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        logger.warning("进入演示模式")
        DEMO_MODE = True
        return None

def validate_file(file):
    """验证上传的文件
    
    Args:
        file: 上传的文件对象
        
    Returns:
        tuple: (是否有效, 错误信息)
    """
    # 检查文件是否存在
    if not file:
        return False, "没有上传文件"
        
    # 检查文件名
    if file.filename == '':
        return False, "未选择文件"
        
    # 检查文件类型
    if not allowed_file(file.filename):
        return False, "不支持的文件类型"
        
    # 检查文件大小（5MB限制）
    if len(file.read()) > 5 * 1024 * 1024:
        return False, "文件大小超过5MB限制"
    file.seek(0)  # 重置文件指针
    
    return True, None

def generate_demo_predictions():
    """生成演示模式下的预测结果"""
    # 随机选择3个类别
    selected_classes = random.sample(CLASSES, 3)
    
    # 生成随机概率
    probabilities = []
    total = 0
    for _ in range(2):
        p = random.uniform(0, 1 - total)
        probabilities.append(p)
        total += p
    probabilities.append(1 - total)
    
    # 排序概率（降序）
    probabilities.sort(reverse=True)
    
    # 创建预测结果
    predictions = [
        {
            'class': cls,
            'probability': prob * 100
        }
        for cls, prob in zip(selected_classes, probabilities)
    ]
    
    return predictions

@api.route('/api/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    if 'image' not in request.files:
        return jsonify({
            'status': 400,
            'error': '没有上传图片',
            'inference_time': 0
        }), 400
        
    file = request.files['image']
    
    # 验证文件
    is_valid, error_message = validate_file(file)
    if not is_valid:
        return jsonify({
            'status': 400,
            'error': error_message,
            'inference_time': 0
        }), 400
        
    try:
        # 保存上传的图片
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # 加载模型（如果尚未加载）
        model = load_model()
        
        # 演示模式
        if DEMO_MODE:
            # 模拟处理时间
            time.sleep(0.5)
            
            # 生成演示预测结果
            results = generate_demo_predictions()
            
            inference_time = (time.time() - start_time) * 1000
            logger.info(f"演示模式预测完成，耗时 {inference_time:.2f}ms")
            
            return jsonify({
                'status': 200,
                'data': {
                    'predictions': results
                },
                'inference_time': round(inference_time, 2),
                'demo_mode': True
            })
        
        # 正常模式
        # 加载和预处理图片
        image = Image.open(temp_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # 获取top-3预测结果
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        results = [
            {
                'class': CLASSES[idx.item()],
                'probability': float(prob.item()) * 100
            }
            for prob, idx in zip(top3_prob, top3_indices)
        ]
        
        # 清理临时文件
        os.remove(temp_path)
        
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        logger.info(f"预测完成，耗时 {inference_time:.2f}ms，结果: {results[0]['class']}")
        
        return jsonify({
            'status': 200,
            'data': {
                'predictions': results
            },
            'inference_time': round(inference_time, 2),
            'demo_mode': False
        })
        
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        # 确保清理临时文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        
        # 如果出错，返回演示模式的结果
        results = generate_demo_predictions()
        inference_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'status': 200,
            'data': {
                'predictions': results
            },
            'inference_time': round(inference_time, 2),
            'demo_mode': True,
            'error_message': str(e)
        }) 