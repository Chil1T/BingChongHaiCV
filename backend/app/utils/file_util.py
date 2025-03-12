import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """检查文件是否为允许的类型"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model_path():
    """获取模型文件路径
    
    返回训练好的最佳模型路径，优先使用环境变量中指定的路径，
    然后尝试best_model.pth, last_model.pth, plant_disease_model.pth
    """
    try:
        # 首先尝试从环境变量获取模型路径
        env_model_path = os.getenv('MODEL_PATH')
        
        # 获取项目根目录
        if 'PYTHONPATH' in os.environ:
            root_dir = os.environ['PYTHONPATH'].split(os.pathsep)[0]
            logger.info(f"从环境变量获取项目根目录: {root_dir}")
        else:
            # 如果环境变量中没有，则使用相对路径
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            logger.info(f"从相对路径获取项目根目录: {root_dir}")
        
        # 如果环境变量中指定了模型路径
        if env_model_path:
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(env_model_path):
                full_model_path = os.path.join(root_dir, env_model_path)
            else:
                full_model_path = env_model_path
                
            logger.info(f"从环境变量获取模型路径: {full_model_path}")
            
            # 检查文件是否存在
            if os.path.exists(full_model_path):
                logger.info(f"找到环境变量指定的模型: {full_model_path}")
                return full_model_path
            else:
                logger.warning(f"环境变量指定的模型文件不存在: {full_model_path}")
        
        # 模型目录
        model_dir = os.path.join(root_dir, 'models')
        logger.info(f"模型目录: {model_dir}")
        
        # 检查目录是否存在
        if not os.path.exists(model_dir):
            logger.warning(f"模型目录不存在，创建目录: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
        
        # 优先使用best_model.pth
        best_model_path = os.path.join(model_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            logger.info(f"找到最佳模型: {best_model_path}")
            return best_model_path
        
        # 如果best_model.pth不存在，尝试使用last_model.pth
        last_model_path = os.path.join(model_dir, 'last_model.pth')
        if os.path.exists(last_model_path):
            logger.info(f"找到最后一个模型: {last_model_path}")
            return last_model_path
        
        # 如果都不存在，尝试使用默认模型
        default_model_path = os.path.join(model_dir, 'plant_disease_model.pth')
        if os.path.exists(default_model_path):
            logger.info(f"找到默认模型: {default_model_path}")
            return default_model_path
        
        # 如果没有找到任何模型，返回None
        logger.error(f"找不到模型文件。请确保在 {model_dir} 目录下存在 best_model.pth, last_model.pth 或 plant_disease_model.pth 文件。")
        return None
    
    except Exception as e:
        logger.error(f"获取模型路径时出错: {str(e)}")
        return None

def get_class_names():
    """获取类别名称"""
    return [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry___Powdery_mildew',
        'Cherry___healthy',
        'Corn___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn___Common_rust',
        'Corn___Northern_Leaf_Blight',
        'Corn___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

def is_demo_mode():
    """检查是否为演示模式"""
    return os.getenv('DEMO_MODE', 'False').lower() in ('true', '1', 't') 