FROM python:3.8-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY backend/requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY backend /app/backend
COPY config /app/config
COPY model /app/model

# 设置环境变量
ENV FLASK_APP=backend/app/routes.py
ENV FLASK_ENV=production

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["flask", "run", "--host=0.0.0.0"] 