import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model_path():
    """获取模型文件路径
    
    返回训练好的最佳模型路径，优先使用best_model.pth
    """
    # 获取项目根目录
    try:
        # 尝试从环境变量获取项目根目录
        if 'PYTHONPATH' in os.environ:
            root_dir = os.environ['PYTHONPATH'].split(os.pathsep)[0]
            logger.info(f"从环境变量获取项目根目录: {root_dir}")
        else:
            # 如果环境变量中没有，则使用相对路径
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            logger.info(f"从相对路径获取项目根目录: {root_dir}")
        
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
        
        # 如果没有找到任何模型，使用示例模型
        logger.error(f"找不到模型文件。请确保在 {model_dir} 目录下存在 best_model.pth, last_model.pth 或 plant_disease_model.pth 文件。")
        
        # 创建一个简单的示例模型，用于演示
        logger.warning("将使用示例模型进行演示")
        return None
    
    except Exception as e:
        logger.error(f"获取模型路径时出错: {str(e)}")
        return None

def get_class_names():
    """获取疾病类别名称列表"""
    return [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ] 