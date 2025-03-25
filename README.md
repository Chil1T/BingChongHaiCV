# 植物病虫害智能识别系统 (Release Version)

基于ResNet50的深度学习植物病虫害识别系统，支持38种常见植物病害的高精度识别。本版本为稳定发布版，专注于系统的部署和使用。

## 系统特点

- **高精度识别**: 支持38类植物病害，验证集准确率达99.70%
- **快速响应**: 优化的推理流程，支持实时识别
- **易于部署**: 提供详细安装指南
- **用户友好**: 直观的Web界面，支持拖拽上传和实时预览

## 系统要求

- Python 3.9+
- Node.js (推荐使用16.x或18.x版本)
- 内存 >= 8GB
- 硬盘空间 >= 2GB

## 下载模型文件

在使用系统前，请先下载预训练模型文件：

1. 从以下链接下载模型文件：
   [best_model.pth](https://your-model-hosting-url.com/best_model.pth)
   
2. 将下载的`best_model.pth`文件放置在项目的`models/`目录下

## 安装与启动

### 选项1: 使用一键启动脚本（Windows用户推荐）

```bash
# 以管理员权限运行
start_admin.bat
```

此脚本提供以下选项：
1. 安装依赖并启动应用
2. 仅启动应用（已安装依赖）

### 选项2: 手动安装与启动（适用于所有平台）

#### 使用Conda虚拟环境（推荐）

```bash
# 1. 创建conda环境
conda create -n plant_disease python=3.9 -y
conda activate plant_disease

# 2. 清除可能存在的代理设置
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=

# 3. 安装Python依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装前端依赖
cd frontend
npm install
cd ../simple_frontend
npm install
cd ..
```

#### 使用Docker部署（推荐用于生产环境）

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

#### 使用Python虚拟环境

```bash
# 1. 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 2. 清除可能存在的代理设置
# Linux/macOS
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
# Windows
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=

# 3. 安装Python依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装前端依赖
cd frontend
npm install
# 注意：前端依赖由package.json管理，无需额外的依赖文件

# 5. 返回项目根目录
cd ..
```

### 启动应用

#### 启动后端

```bash
# 确保已激活虚拟环境
cd backend
python app.py
```

#### 启动前端（在另一个终端窗口）

```bash
cd frontend
npm start
```

我们已在前端的package.json中设置了Node.js兼容选项（NODE_OPTIONS=--openssl-legacy-provider），因此npm start应该可以正常运行。

启动后，在浏览器中访问 http://localhost:3000

## 停止应用

在各自的终端窗口中按 `Ctrl+C` 停止前端和后端服务。

## 使用指南

1. 启动应用后，在浏览器中访问 http://localhost:3000
2. 点击上传区域或拖拽图片进行上传（支持JPG、PNG格式，≤5MB）
3. 等待系统识别（通常在1-2秒内完成）
4. 查看识别结果和防治建议

## 支持的病害类型

系统当前支持38种常见植物病害的识别，包括：
- 苹果：黑星病、疮痂病等
- 葡萄：黑腐病、褐斑病等
- 玉米：灰斑病、锈病等
- 马铃薯：早疫病、晚疫病等
- 番茄：叶霉病、斑点病等

完整列表请参见[支持的病害类型](docs/supported_diseases.md)

## 项目结构

```
BingChongHaiCV/
├── backend/                # Flask后端服务
│   ├── app/               # 后端应用核心代码
│   ├── app.py             # 后端入口文件
│   ├── requirements.txt   # 后端依赖
│   └── .env              # 后端环境配置
├── frontend/              # React前端应用（主界面）
│   ├── src/              # 前端源代码
│   ├── public/           # 静态资源
│   ├── package.json      # 前端依赖配置
│   └── .env             # 前端环境配置
├── simple_frontend/      # 简化版前端界面
├── model/                # 模型相关代码
├── data/                 # 数据集和训练数据
├── docs/                 # 项目文档
├── scripts/              # 工具脚本
├── evaluation/           # 模型评估代码
├── notebooks/            # Jupyter notebooks
├── logs/                 # 日志文件
├── config/               # 配置文件
├── temp/                 # 临时文件
├── requirements.txt      # 完整Python依赖
├── start_admin.bat       # Windows启动脚本
├── stop_app.bat          # 停止应用脚本
├── docker-compose.yml    # Docker编排配置
└── Dockerfile           # Docker构建文件
```

## 常见问题

### Q: 系统提示"模型文件不存在"怎么办？

A: 请确保已下载模型文件并正确放置在`models/`目录下。

### Q: 前端启动时出现"digital envelope routines::unsupported"错误怎么办？

A: 这是由于Node.js版本过高导致的。我们已在package.json中添加了兼容设置，但如果仍有问题，可以：

1. 使用环境变量启动：
   ```bash
   # Windows
   set NODE_OPTIONS=--openssl-legacy-provider
   npm start
   
   # Linux/macOS
   export NODE_OPTIONS=--openssl-legacy-provider
   npm start
   ```

2. 使用兼容版本的Node.js (v16.x)：
   ```bash
   # 安装nvm后切换Node.js版本
   nvm install 16
   nvm use 16
   ```

### Q: 启动时提示"端口被占用"怎么办？

A: 请先检查是否有其他应用占用了相关端口：
1. 后端默认使用5000端口，前端默认使用3000端口
2. 可以通过修改后端的app.py文件来更改端口
3. 对于前端，npm start时会自动提示是否使用其他端口

### Q: 依赖安装失败或网络问题怎么办？

A: 尝试以下解决方案：
```bash
# 完全清除代理设置
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
npm config set registry https://registry.npmmirror.com
```

### Q: 后端的requirements.txt和根目录的requirements.txt有什么区别？

A: 为简化安装流程，我们已将后端的requirements.txt内容合并到根目录的requirements.txt中。现在只需安装根目录的requirements.txt即可运行整个项目。

## 技术支持

如遇问题，请：
1. 查看[常见问题文档](docs/faq.md)
2. 提交[Issue](https://github.com/Chil1T/BingChongHaiCV/issues)
3. 发送邮件至技术支持邮箱

## 更新日志

### v1.0.1 (2024-03-25)
- 添加简化版前端界面
- 优化项目结构，完善文档
- 添加Docker部署支持
- 修复已知bug

### v1.0.0 (2024-03-12)
- 首个稳定发布版本
- 优化模型加载和推理性能
- 改进用户界面交互体验
- 增加详细的部署文档

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 
