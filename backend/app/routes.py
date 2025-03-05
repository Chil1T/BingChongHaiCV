from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import logging
from logging.handlers import RotatingFileHandler
import clamd
from .utils.file_util import allowed_file, get_model_path

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

# 初始化ClamAV客户端
try:
    clamav = clamd.ClamdUnixSocket()
    logger.info("ClamAV service initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize ClamAV: {e}")
    clamav = None

# 加载模型
def load_model():
    model_path = get_model_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 疾病类别
CLASSES = [
    "健康",
    "叶斑病",
    "炭疽病",
    "白粉病",
    "锈病",
    "虫害",
    "病毒病",
    "细菌性病害",
    "营养缺乏",
    "其他"
]

def scan_file(file_path):
    """扫描文件是否包含病毒"""
    if clamav is None:
        logger.warning("ClamAV is not available, skipping virus scan")
        return True
    
    try:
        scan_result = clamav.scan(file_path)
        is_clean = scan_result[file_path][0] == 'OK'
        if not is_clean:
            logger.warning(f"Virus detected in file: {file_path}")
        return is_clean
    except Exception as e:
        logger.error(f"Error during virus scan: {e}")
        return False

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
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({
            'status': 400,
            'error': '无效的文件类型',
            'inference_time': 0
        }), 400
        
    try:
        # 保存上传的图片
        filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        # 病毒扫描
        if not scan_file(temp_path):
            os.remove(temp_path)
            return jsonify({
                'status': 400,
                'error': '文件可能包含病毒',
                'inference_time': 0
            }), 400
        
        # 加载和预处理图片
        image = Image.open(temp_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # 预测
        model = load_model()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # 获取top-3预测结果
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        results = [
            {
                'class': CLASSES[idx],
                'probability': float(prob) * 100
            }
            for prob, idx in zip(top3_prob, top3_indices)
        ]
        
        # 清理临时文件
        os.remove(temp_path)
        
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        logger.info(f"Prediction completed in {inference_time:.2f}ms")
        
        return jsonify({
            'status': 200,
            'data': {
                'predictions': results
            },
            'inference_time': round(inference_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'status': 500,
            'error': str(e),
            'inference_time': 0
        }), 500 