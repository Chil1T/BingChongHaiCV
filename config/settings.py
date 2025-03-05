import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Flask配置
FLASK_CONFIG = {
    'SECRET_KEY': os.getenv('SECRET_KEY', 'your-secret-key-here'),
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 最大16MB上传
}

# 深度学习模型配置
MODEL_CONFIG = {
    'IMAGE_SIZE': (224, 224),
    'BATCH_SIZE': 32,
    'NUM_CLASSES': 10,
    'LEARNING_RATE': 0.001,
}

# 跨域配置
CORS_CONFIG = {
    'CORS_ORIGINS': [
        'http://localhost:3000',  # React开发服务器
        'http://localhost:5000',  # Flask开发服务器
    ],
    'CORS_METHODS': ['GET', 'POST', 'OPTIONS'],
    'CORS_ALLOW_HEADERS': ['Content-Type'],
}

# 文件上传配置
UPLOAD_CONFIG = {
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg'},
    'MAX_FILE_SIZE': 5 * 1024 * 1024,  # 5MB
}

# 创建必要的目录
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True) 