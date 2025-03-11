from flask import Flask
from flask_cors import CORS
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入API蓝图
from backend.app.routes import api

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用CORS，允许前端跨域访问

# 注册蓝图
app.register_blueprint(api)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 