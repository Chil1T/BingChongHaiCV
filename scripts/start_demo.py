#!/usr/bin/env python
"""
植物病虫害识别系统启动脚本
用于启动前端和后端服务
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading
import logging
import platform
import shutil
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()

def check_model_exists():
    """检查模型文件是否存在"""
    model_path = os.path.join(ROOT_DIR, 'models', 'best_model.pth')
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        logger.info("请确保已将训练好的模型文件放置在models目录下")
        return False
    return True

def check_command_exists(command):
    """检查命令是否存在"""
    return shutil.which(command) is not None

def start_backend():
    """启动后端服务"""
    logger.info("正在启动后端服务...")
    
    # 创建临时目录
    os.makedirs(os.path.join(ROOT_DIR, 'temp'), exist_ok=True)
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT_DIR)  # 添加项目根目录到Python路径
    
    # 检查Python是否存在
    python_cmd = sys.executable
    if not python_cmd:
        logger.error("找不到Python解释器")
        return None
    
    # 直接运行app.py
    backend_cmd = [python_cmd, os.path.join(ROOT_DIR, 'backend', 'app.py')]
    try:
        backend_process = subprocess.Popen(
            backend_cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except Exception as e:
        logger.error(f"启动后端服务失败: {str(e)}")
        return None
    
    # 等待服务启动
    time.sleep(2)
    
    # 检查服务是否成功启动
    if backend_process.poll() is not None:
        logger.error("后端服务启动失败")
        output, _ = backend_process.communicate()
        logger.error(f"错误信息: {output}")
        return None
    
    logger.info("后端服务已启动，监听端口: 5000")
    return backend_process

def start_frontend_with_python():
    """使用Python的http.server模块启动一个简单的前端服务"""
    logger.info("正在使用Python启动简易前端服务...")
    
    # 创建一个简单的HTML文件，用于访问后端API
    html_dir = os.path.join(ROOT_DIR, 'simple_frontend')
    os.makedirs(html_dir, exist_ok=True)
    
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>植物病虫害识别系统 - 简易版</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; text-align: center; }
            .upload-container { border: 2px dashed #3498db; padding: 20px; text-align: center; margin: 20px 0; }
            .result-container { margin-top: 20px; display: none; }
            .prediction { margin: 10px 0; padding: 10px; border: 1px solid #eee; }
            .prediction-bar { height: 20px; background-color: #2ecc71; margin-top: 5px; }
            img { max-width: 100%; max-height: 300px; margin: 10px 0; }
            .loading { display: none; }
            .demo-mode { background-color: #f8d7da; color: #721c24; padding: 10px; margin: 10px 0; border-radius: 4px; display: none; }
            .error-message { background-color: #f8d7da; color: #721c24; padding: 10px; margin: 10px 0; border-radius: 4px; display: none; }
        </style>
    </head>
    <body>
        <h1>植物病虫害识别系统</h1>
        <div class="upload-container">
            <h2>上传植物图片</h2>
            <input type="file" id="image-upload" accept="image/*">
            <p>支持JPG、PNG格式，文件小于5MB</p>
            <div class="loading">处理中...</div>
        </div>
        <div class="demo-mode" id="demo-mode">
            <strong>演示模式:</strong> 当前系统运行在演示模式下，显示的结果为随机生成，不代表实际识别结果。请确保模型文件已正确放置在models目录下。
        </div>
        <div class="error-message" id="error-message"></div>
        <div class="result-container" id="result-container">
            <h2>识别结果</h2>
            <div id="image-preview"></div>
            <div id="inference-time"></div>
            <div id="predictions"></div>
        </div>
        
        <script>
            document.getElementById('image-upload').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (!file) return;
                
                // 检查文件类型
                if (!file.type.match('image.*')) {
                    alert('请上传图片文件');
                    return;
                }
                
                // 检查文件大小
                if (file.size > 5 * 1024 * 1024) {
                    alert('文件大小不能超过5MB');
                    return;
                }
                
                // 显示加载中
                document.querySelector('.loading').style.display = 'block';
                
                // 创建FormData
                const formData = new FormData();
                formData.append('image', file);
                
                // 发送请求
                fetch('http://localhost:5000/api/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // 隐藏加载中
                    document.querySelector('.loading').style.display = 'none';
                    
                    // 显示结果容器
                    document.getElementById('result-container').style.display = 'block';
                    
                    // 显示图片预览
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const img = document.createElement('img');
                        img.src = e.target.result;
                        const preview = document.getElementById('image-preview');
                        preview.innerHTML = '';
                        preview.appendChild(img);
                    }
                    reader.readAsDataURL(file);
                    
                    // 显示推理时间
                    document.getElementById('inference-time').innerHTML = `<p>推理时间: ${data.inference_time} 毫秒</p>`;
                    
                    // 显示预测结果
                    const predictionsContainer = document.getElementById('predictions');
                    predictionsContainer.innerHTML = '';
                    
                    data.data.predictions.forEach(prediction => {
                        const predDiv = document.createElement('div');
                        predDiv.className = 'prediction';
                        
                        const className = prediction.class.replace(/_/g, ' ').replace(/___/g, ' - ');
                        const probability = prediction.probability.toFixed(2);
                        
                        predDiv.innerHTML = `
                            <div style="display: flex; justify-content: space-between;">
                                <span>${className}</span>
                                <span>${probability}%</span>
                            </div>
                            <div class="prediction-bar" style="width: ${probability}%"></div>
                        `;
                        
                        predictionsContainer.appendChild(predDiv);
                    });
                    
                    // 显示演示模式标志
                    if (data.demo_mode) {
                        document.getElementById('demo-mode').style.display = 'block';
                    } else {
                        document.getElementById('demo-mode').style.display = 'none';
                    }
                    
                    // 显示错误信息
                    if (data.error_message) {
                        const errorElement = document.getElementById('error-message');
                        errorElement.textContent = `错误: ${data.error_message}`;
                        errorElement.style.display = 'block';
                    } else {
                        document.getElementById('error-message').style.display = 'none';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.querySelector('.loading').style.display = 'none';
                    alert('识别失败，请重试');
                });
            });
        </script>
    </body>
    </html>
    """
    
    with open(os.path.join(html_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 启动Python的HTTP服务器
    python_cmd = sys.executable
    server_cmd = [python_cmd, '-m', 'http.server', '3000', '--directory', html_dir]
    
    try:
        frontend_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except Exception as e:
        logger.error(f"启动简易前端服务失败: {str(e)}")
        return None
    
    # 等待服务启动
    time.sleep(2)
    
    # 检查服务是否成功启动
    if frontend_process.poll() is not None:
        logger.error("简易前端服务启动失败")
        output, _ = frontend_process.communicate()
        logger.error(f"错误信息: {output}")
        return None
    
    logger.info("简易前端服务已启动，监听端口: 3000")
    return frontend_process

def start_frontend():
    """启动前端服务"""
    logger.info("正在启动前端服务...")
    
    # 检查npm是否存在
    if not check_command_exists('npm'):
        logger.warning("找不到npm命令，无法启动React前端服务")
        logger.info("将使用Python启动简易前端服务作为替代")
        return start_frontend_with_python()
    
    # 设置环境变量
    env = os.environ.copy()
    env['REACT_APP_API_BASE_URL'] = 'http://localhost:5000'
    
    # 检查前端目录是否存在
    frontend_dir = os.path.join(ROOT_DIR, 'frontend')
    if not os.path.exists(frontend_dir):
        logger.error(f"前端目录不存在: {frontend_dir}")
        logger.info("将使用Python启动简易前端服务作为替代")
        return start_frontend_with_python()
    
    # 检查package.json是否存在
    if not os.path.exists(os.path.join(frontend_dir, 'package.json')):
        logger.error(f"package.json不存在: {os.path.join(frontend_dir, 'package.json')}")
        logger.info("将使用Python启动简易前端服务作为替代")
        return start_frontend_with_python()
    
    # 启动React开发服务器
    frontend_cmd = ['npm', 'start']
    try:
        frontend_process = subprocess.Popen(
            frontend_cmd,
            cwd=frontend_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except Exception as e:
        logger.error(f"启动前端服务失败: {str(e)}")
        logger.info("将使用Python启动简易前端服务作为替代")
        return start_frontend_with_python()
    
    # 等待服务启动
    time.sleep(5)
    
    # 检查服务是否成功启动
    if frontend_process.poll() is not None:
        logger.error("前端服务启动失败")
        output, _ = frontend_process.communicate()
        logger.error(f"错误信息: {output}")
        logger.info("将使用Python启动简易前端服务作为替代")
        return start_frontend_with_python()
    
    logger.info("前端服务已启动，监听端口: 3000")
    return frontend_process

def monitor_process(process, name):
    """监控进程输出"""
    for line in iter(process.stdout.readline, ''):
        logger.info(f"[{name}] {line.strip()}")
    
    logger.warning(f"{name}进程已退出，退出码: {process.returncode}")

def main():
    """主函数"""
    logger.info("植物病虫害识别系统启动脚本")
    
    # 检查模型文件
    if not check_model_exists():
        return
    
    # 启动后端服务
    backend_process = start_backend()
    if backend_process is None:
        return
    
    # 启动前端服务
    frontend_process = start_frontend()
    if frontend_process is None:
        backend_process.terminate()
        return
    
    # 创建监控线程
    backend_monitor = threading.Thread(
        target=monitor_process,
        args=(backend_process, "后端"),
        daemon=True
    )
    frontend_monitor = threading.Thread(
        target=monitor_process,
        args=(frontend_process, "前端"),
        daemon=True
    )
    
    # 启动监控线程
    backend_monitor.start()
    frontend_monitor.start()
    
    # 打开浏览器
    logger.info("正在打开浏览器...")
    webbrowser.open('http://localhost:3000')
    
    logger.info("系统已启动，按Ctrl+C停止服务")
    
    try:
        # 等待用户中断
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("正在停止服务...")
    finally:
        # 停止服务
        frontend_process.terminate()
        backend_process.terminate()
        logger.info("服务已停止")

if __name__ == "__main__":
    main() 