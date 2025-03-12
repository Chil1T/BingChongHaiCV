# 植物病虫害智能识别系统 (Release Version)

基于ResNet50&kaggle plantvillage dataset color的深度学习植物病虫害识别系统，支持38种常见植物病害的高精度识别。本版本为预览发布版，专注于系统的部署和使用。
详细模型的训练数据可见[技术报告](https://github.com/Chil1T/BingChongHaiCV/blob/release/docs/technical_report.md)

## 系统特点

- **高精度识别**: 支持38类植物病害，验证集准确率达99.70%
- **快速响应**: 优化的推理流程，支持实时识别
- **易于部署**: 提供一键安装脚本，支持Windows/Linux/macOS
- **用户友好**: 直观的Web界面，支持拖拽上传和实时预览

## 快速开始

### 系统要求

- Python 3.7+
- 内存 >= 8GB
- 硬盘空间 >= 2GB

### 下载模型文件+clone仓库release分支/下载release界面压缩包直接运行

#### 下载预训练模型文件：

1. 从以下链接下载模型文件：
   [best_model.pth](https://github.com/Chil1T/BingChongHaiCV/releases/tag/%E6%A8%A1%E5%9E%8B%E6%96%87%E4%BB%B6)
   
2. clone 仓库 release分支后，将下载的`best_model.pth`文件放置在项目的`models/`目录下
> 如果没有请自行创建一个
#### 或者  直接下载压缩包
[BCHCVrelease.zip](https://github.com/Chil1T/BingChongHaiCV/releases/tag/%E9%A1%B9%E7%9B%AE)
> 解压后即可开始安装与启动
### 安装与启动

#### Windows用户（推荐）

```bash
# 以管理员权限运行
start_admin.bat
```

此脚本提供以下选项：
1. 安装依赖并启动应用
2. 仅启动应用（已安装依赖）

#### 其他平台用户
> 暂未测试，可能遇到依赖版本冲突

```bash
# 1. 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
python start_app.py
```

### 停止应用

- Windows: 运行 `stop_app.bat`
- 其他平台: 在终端中按 `Ctrl+C`

## 使用指南

1. 启动应用后，在浏览器中访问 http://localhost:5000
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



## 项目结构

```
BingChongHaiCV/
├── backend/                # Flask后端服务
├── frontend/               # React前端应用
├── models/                 # 模型文件目录（需手动下载模型）
├── requirements.txt        # Python依赖
├── start_admin.bat        # Windows启动脚本
└── stop_app.bat           # Windows停止脚本
```

## 常见问题

### Q: 系统提示"模型文件不存在"怎么办？

A: 请确保已下载模型文件并正确放置在`models/`目录下。

### Q: 启动时提示"端口被占用"怎么办？

A: 请先检查是否有其他应用占用了5000端口，可以：
1. 运行`stop_app.bat`关闭已有服务
2. 手动结束占用端口的进程

### Q: 识别结果不准确怎么办？

A: 请确保：
1. 上传的图片清晰、对焦准确
2. 病害部位在图片中清晰可见
3. 图片中的植物属于支持识别的类型

### Q: 依赖安装失败怎么办？

A: 建议使用国内镜像源加速安装：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 技术支持

如遇问题，请：
1. 提交[Issue](https://github.com/Chil1T/BingChongHaiCV/issues)
2. 发送邮件至技术支持邮箱 1379928025@qq.com


## 更新日志

### v1.0.0 (2024-03-12)
- 首个稳定发布版本
- 优化模型加载和推理性能
- 改进用户界面交互体验
- 增加详细的部署文档

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 
