# 植物病虫害图像识别系统

基于深度学习的植物病虫害图像识别工具，使用PyTorch和React构建。

## 功能特点

- 支持多种常见植物病虫害的识别
- 实时图像上传和预测
- 友好的用户界面
- Docker容器化部署
- RESTful API设计

## 技术栈

### 后端
- Python 3.8
- Flask
- PyTorch
- Pillow
- NumPy

### 前端
- React
- TypeScript
- Ant Design
- Axios

## 快速开始

### 使用Docker Compose（推荐）

1. 克隆项目：
```bash
git clone <repository-url>
cd plant-disease-detection
```

2. 启动服务：
```bash
docker-compose up --build
```

3. 访问应用：
- 前端界面：http://localhost:3000
- API接口：http://localhost:5000

### 手动安装

1. 安装后端依赖：
```bash
cd backend
pip install -r requirements.txt
```

2. 安装前端依赖：
```bash
cd frontend
npm install
```

3. 启动后端服务：
```bash
cd backend
flask run
```

4. 启动前端服务：
```bash
cd frontend
npm start
```

## 项目结构

```
├── model/                 # 深度学习模块
│   ├── train.py          # 模型训练脚本
│   └── dataloader/       # 数据加载器
├── frontend/             # 前端模块
│   ├── src/
│   │   ├── components/   # React组件
│   │   └── api/         # API客户端
│   └── public/
├── backend/              # 后端模块
│   ├── app/             
│   │   ├── routes.py     # API路由
│   │   └── utils/       # 工具函数
│   └── requirements.txt  # Python依赖
└── config/               # 配置文件
```

## API文档

### POST /api/predict
上传图片并获取预测结果

请求：
- Content-Type: multipart/form-data
- Body: image文件

响应：
```json
{
  "predictions": [
    {
      "class": "病虫害类型",
      "probability": 95.5
    },
    ...
  ]
}
```

## 模型训练

1. 准备数据集：
   - 将训练图像放在 `data/train` 目录
   - 将验证图像放在 `data/val` 目录

2. 开始训练：
```bash
python model/train.py
```

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交改动
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License 