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
from backend.app.utils.file_util import allowed_file, get_model_path, get_class_names, is_demo_mode

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
DEMO_MODE = is_demo_mode()  # 从环境变量读取演示模式配置

logger.info(f"演示模式: {DEMO_MODE}")

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
        logger.info("模型已加载，直接返回")
        return model
    
    try:
        model_path = get_model_path()
        logger.info(f"尝试加载模型: {model_path}")
        
        # 如果没有找到模型文件，进入演示模式
        if model_path is None or not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            DEMO_MODE = True
            return None
        
        # 创建模型架构
        logger.info("创建ResNet50模型架构")
        model = models.resnet50(pretrained=False)
        num_classes = len(CLASSES)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        logger.info(f"模型输出层已调整为{num_classes}个类别")
        
        try:
            # 加载模型权重
            logger.info(f"开始加载模型权重: {model_path}")
            logger.info(f"当前设备: {device}")
            
            # 使用map_location确保模型加载到正确的设备上
            checkpoint = torch.load(model_path, map_location=device)
            logger.info(f"成功加载checkpoint: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    logger.info("从checkpoint中提取model_state_dict")
                    state_dict = checkpoint["model_state_dict"]
                else:
                    logger.info("直接使用checkpoint作为state_dict")
                    state_dict = checkpoint
                
                # 检查state_dict中的键是否与模型匹配
                model_keys = set(model.state_dict().keys())
                state_dict_keys = set(state_dict.keys())
                
                # 如果键不完全匹配，尝试调整
                if model_keys != state_dict_keys:
                    logger.warning("模型键与state_dict键不匹配，尝试调整")
                    
                    # 检查是否有module前缀问题
                    if all(k.startswith('module.') for k in state_dict_keys):
                        logger.info("检测到module前缀，移除中...")
                        new_state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
                        state_dict = new_state_dict
                    
                    # 检查是否缺少键
                    missing_keys = model_keys - set(state_dict.keys())
                    if missing_keys:
                        logger.warning(f"模型中缺少的键: {missing_keys}")
                    
                    # 检查是否有多余的键
                    extra_keys = set(state_dict.keys()) - model_keys
                    if extra_keys:
                        logger.warning(f"state_dict中多余的键: {extra_keys}")
                        # 移除多余的键
                        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
                
                # 加载调整后的state_dict
                logger.info("加载state_dict到模型")
                model.load_state_dict(state_dict, strict=False)
                logger.info("成功加载模型权重")
            else:
                logger.error(f"无效的checkpoint格式: {type(checkpoint)}")
                DEMO_MODE = True
                return None
            
            model.to(device)
            model.eval()
            logger.info(f"模型加载成功，类别数: {num_classes}，设备: {device}")
            return model
            
        except Exception as e:
            logger.error(f"加载模型权重失败: {str(e)}")
            logger.exception("详细错误信息:")
            DEMO_MODE = True
            return None
            
    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        logger.exception("详细错误信息:")
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
        try:
            # 检查NumPy是否可用
            import numpy as np
            logger.info("NumPy版本: " + np.__version__)
            
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
        except ImportError as e:
            logger.error(f"NumPy导入错误: {str(e)}")
            logger.error("这可能是由于NumPy 2.0与PyTorch的兼容性问题导致的，请降级NumPy到1.24.x版本")
            # 如果NumPy不可用，使用演示模式
            results = generate_demo_predictions()
            inference_time = (time.time() - start_time) * 1000
            
            return jsonify({
                'status': 200,
                'data': {
                    'predictions': results
                },
                'inference_time': round(inference_time, 2),
                'demo_mode': True,
                'error_message': "NumPy兼容性错误，请运行fix_numpy.bat修复"
            })
        
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        logger.exception("详细错误信息:")
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

@api.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        # 检查NumPy兼容性
        numpy_compatible = True
        numpy_version = None
        numpy_error = None
        
        try:
            import numpy as np
            numpy_version = np.__version__
            logger.info(f"健康检查: NumPy版本 {numpy_version}")
            
            # 检查NumPy版本是否与PyTorch兼容
            major, minor, patch = map(int, numpy_version.split('.'))
            if major > 1:
                numpy_compatible = False
                numpy_error = f"NumPy版本 {numpy_version} 可能与PyTorch不兼容，建议降级到1.24.x"
                logger.warning(numpy_error)
        except ImportError as e:
            numpy_compatible = False
            numpy_error = str(e)
            logger.error(f"NumPy导入错误: {numpy_error}")
        
        # 主动尝试加载模型（如果尚未加载）
        if model is None and not DEMO_MODE:
            logger.info("健康检查时尝试加载模型")
            # 使用线程加载模型，避免阻塞请求
            import threading
            threading.Thread(target=load_model, daemon=True).start()
        
        # 检查模型是否已加载
        model_loaded = model is not None
        model_path = get_model_path()
        model_exists = model_path is not None and os.path.exists(model_path)
        
        # 获取模型文件大小
        model_size = None
        if model_exists:
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        # 如果模型文件存在但模型未加载，返回加载中状态
        loading_in_progress = model_exists and not model_loaded and not DEMO_MODE
        
        # 确定演示模式的原因
        demo_mode_reason = None
        if DEMO_MODE:
            if not model_exists:
                demo_mode_reason = '模型文件不存在'
            elif not numpy_compatible:
                demo_mode_reason = f'NumPy兼容性问题: {numpy_error}'
            else:
                demo_mode_reason = '模型加载失败'
        
        return jsonify({
            'status': 'healthy',
            'mode': 'demo' if DEMO_MODE else 'production',
            'model_loaded': model_loaded,
            'loading_in_progress': loading_in_progress,
            'numpy_info': {
                'version': numpy_version,
                'compatible': numpy_compatible,
                'error': numpy_error
            },
            'model_info': {
                'path': model_path,
                'exists': model_exists,
                'size_mb': round(model_size, 2) if model_size else None,
                'device': str(device)
            },
            'demo_mode_reason': demo_mode_reason
        })
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500 