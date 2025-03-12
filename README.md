# 植物病虫害智能识别系统

基于ResNet50的深度学习植物病虫害识别系统，支持38种常见植物病害的高精度识别。

## 研究价值

植物病虫害每年导致全球约40%的作物减产。本研究通过深度学习实现快速、准确的自动识别，具有显著的生产、环境和经济价值。

## 核心技术

- **数据**：PlantVillage数据集（54,305张图像，38类病害）
- **模型**：ResNet50（残差连接、预训练权重）
- **训练**：AdamW优化、余弦退火学习率、早停策略、混合精度

## 实验结果

- **准确率**：训练集99.90%，验证集99.70%
- **效率**：1.2小时完成训练（T4 GPU）
- **关键发现**：
  1. 模型部分依赖背景信息进行识别
  2. 存在一定过拟合但保持良好泛化能力

## 系统实现

- **前端**：React UI，实时预览，结果可视化
- **后端**：Flask API，高效推理，完善错误处理

## 未来方向

- **架构优化**：探索Vision Transformer、引入注意力机制
- **数据策略**：混合训练、真实场景增强、对比学习
- **应用拓展**：移动部署、多模态融合、预警系统

## 快速开始

### 环境要求

- Python 3.7+
- Node.js 14+
- npm 6+

### 一键安装与启动

我们提供了一键安装与启动脚本，可以自动完成依赖安装和应用启动：

#### Windows用户（推荐）

```bash
# 管理员权限运行
start_admin.bat
```

此脚本会提供以下选项：
1. 安装依赖并启动应用
   - 自动检测依赖是否已安装，避免重复安装
   - 可选择国内镜像源或官方源
2. 仅启动应用（已安装依赖）

#### 单独安装依赖（如遇安装问题）

如果一键启动脚本无法正确安装依赖，可以使用专门的依赖安装脚本：

```bash
# 管理员权限运行
install_deps.bat
```

此脚本专注于依赖安装，提供详细的安装步骤和错误反馈。

#### 修复版安装脚本（推荐）

如果您在安装过程中遇到pip升级错误或其他问题，请使用修复版安装脚本：

```bash
# 管理员权限运行
install_deps_fixed.bat
```

此修复版脚本具有以下优势：
- 更强的错误处理能力，即使某些依赖安装失败也会继续安装其他依赖
- 跳过pip升级错误检查，避免常见的升级失败问题
- 提供更详细的安装进度和错误反馈
- 在安装结束时验证关键依赖是否成功安装

#### 所有平台（Windows/macOS/Linux）

```bash
# 跨平台启动脚本
python start_app.py
```

此脚本会自动检测操作系统，并使用适当的命令启动应用程序。

### 停止应用程序

当您需要停止应用程序时，可以使用以下方法：

#### Windows用户

```bash
# 停止应用程序
stop_app.bat
```

#### 其他平台

在运行应用程序的终端中按 `Ctrl+C` 停止服务。

## 使用说明

1. 上传植物图片（支持JPG、PNG格式，大小不超过5MB）
2. 系统会自动识别图片中的植物病害
3. 查看识别结果和防治建议
4. 点击"支持疾病类型"查看系统支持的所有植物病害信息
5. 点击"识别历史"查看历史识别记录

## 项目结构

```
BingChongHaiCV/
├── backend/                # 后端服务
│   ├── app/                # Flask应用
│   │   ├── __init__.py     # 应用初始化
│   │   ├── routes.py       # API路由
│   │   └── utils/          # 工具函数
│   ├── app.py              # 应用入口
│   └── models/             # 模型定义
├── frontend/               # 前端应用
│   ├── public/             # 静态资源
│   ├── src/                # 源代码
│   │   ├── components/     # React组件
│   │   ├── App.tsx         # 应用入口
│   │   └── index.tsx       # 渲染入口
│   ├── package.json        # 依赖配置
│   └── tsconfig.json       # TypeScript配置
├── datasets/               # 数据集（本地存储，不提交到Git）
├── models/                 # 训练好的模型（本地存储，不提交到Git）
├── notebooks/              # Jupyter笔记本
├── scripts/                # 工具脚本
├── requirements.txt        # Python依赖
├── .gitignore              # Git忽略配置
├── README.md               # 项目说明
├── start_admin.bat         # Windows一键安装与启动脚本（管理员权限）
├── install_deps.bat        # Windows专用依赖安装脚本（管理员权限）
├── stop_app.bat            # Windows停止应用脚本
└── start_app.py            # 跨平台启动脚本
```

## 常见问题

### Q: 系统提示"服务正在初始化"怎么办？

A: 后端服务需要一些时间加载模型，请耐心等待。如果长时间未响应，请检查后端服务是否正常运行。

### Q: 上传图片后提示"识别失败"怎么办？

A: 请确保上传的是清晰的植物图片，并且图片格式和大小符合要求。如果问题持续，可能是后端服务未正确启动。

### Q: 如何关闭服务？

A: 如果使用一键启动脚本，可以按Ctrl+C关闭服务。如果手动启动，需要分别关闭前端和后端的命令行窗口。

### Q: 遇到npm权限错误怎么办？

A: 如果遇到npm权限错误（EPERM: operation not permitted），请按以下步骤操作：
1. 关闭所有命令行窗口
2. 运行 `clean_npm_locks.bat` 清理npm锁定文件
3. 使用 `start_app_admin.bat` 以管理员权限启动应用

### Q: 遇到PyTorch安装错误怎么办？

A: 如果遇到PyTorch安装错误，可能是版本不兼容。请尝试以下解决方案：
1. 修改 `backend/requirements.txt` 文件，使用CPU版本的PyTorch
2. 手动安装PyTorch：`pip install torch==2.0.1 torchvision==0.15.2`

### Q: 依赖安装很慢怎么办？

A: 国内网络环境下依赖安装可能较慢，建议使用国内镜像源：
1. 运行 `install_dependencies.bat` 或 `install_dependencies_admin.bat`（推荐）脚本
2. 这些脚本会自动使用清华PyPI镜像和淘宝NPM镜像加速安装

### Q: 手动配置国内镜像源的方法？

A: 如果需要手动配置国内镜像源，可以使用以下命令：

**Python/pip镜像源**:
```bash
# 临时使用
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 永久设置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**Node.js/npm镜像源**:
```bash
# 设置淘宝镜像
npm config set registry https://registry.npmmirror.com
npm config set disturl https://npmmirror.com/dist
```

### Q: 启动脚本显示乱码怎么办？

A: 这是由于命令行编码问题导致的。请尝试以下解决方案：
1. 在命令行中运行 `chcp 65001` 切换到UTF-8编码
2. 使用最新版本的启动脚本，已添加编码设置

### Q: 遇到"No module named 'flask'"错误怎么办？

A: 这表明Flask依赖未正确安装。请尝试以下解决方案：
1. 运行`install_deps_fixed.bat`脚本（推荐）专门安装依赖
2. 手动安装Flask：`pip install flask==2.0.1 flask-cors==3.0.10`
3. 确保在虚拟环境中运行：`call venv\Scripts\activate.bat`

### Q: 遇到pip升级错误怎么办？

A: 如果在安装过程中遇到pip升级错误（如"ERROR: To modify pip..."），请尝试以下解决方案：
1. 使用`install_deps_fixed.bat`脚本，它会跳过pip升级错误检查
2. 手动运行pip安装命令：`python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
3. 如果仍然遇到问题，可以尝试在管理员权限的命令提示符中运行：`python -m pip install --upgrade pip`

### Q: 遇到"Numpy is not available"或NumPy兼容性错误怎么办？

A: 这是由于NumPy 2.0与PyTorch等库的兼容性问题导致的。请尝试以下解决方案：

1. 运行NumPy兼容性修复脚本：
```bash
# 管理员权限运行
fix_numpy.bat
```

2. 或者手动降级NumPy到兼容版本：
```bash
# 在虚拟环境中执行
pip uninstall -y numpy
pip install numpy==1.24.4
```

3. 如果问题仍然存在，请尝试重新安装PyTorch：
```bash
pip uninstall -y torch torchvision
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## 技术报告

详细的实验分析和可视化结果请参见[技术报告](docs/technical_report.md)。

## 许可证

MIT License 
