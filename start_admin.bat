@echo off
chcp 65001 > nul

:: 设置错误处理
setlocal EnableDelayedExpansion

:: 检查管理员权限
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo 请求管理员权限...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    pushd "%CD%"
    CD /D "%~dp0"

cls
echo ======================================================
echo 植物病害智能识别系统 - 一键启动脚本 (管理员权限)
echo ======================================================
echo.

echo 请选择操作:
echo 1. 安装依赖并启动应用
echo 2. 仅启动应用
echo.
set /p op_choice="请输入选择 (1/2): "

:: 检查输入是否有效
if not "%op_choice%"=="1" if not "%op_choice%"=="2" (
    echo 错误: 无效的选择，请输入1或2
    goto error_pause
)

if "%op_choice%"=="1" (
    echo.
    
    :: 检查依赖是否已安装
    set DEPS_INSTALLED=0
    if exist "venv" (
        call venv\Scripts\activate.bat 2>nul || (
            echo 错误: 无法激活虚拟环境
            goto error_pause
        )
        python -c "import flask" 2>nul
        if !ERRORLEVEL! equ 0 (
            python -c "import torch" 2>nul
            if !ERRORLEVEL! equ 0 (
                if exist "frontend\node_modules" (
                    set DEPS_INSTALLED=1
                    echo 检测到依赖已安装完成，无需重新安装。
                    echo.
                )
            )
        )
        call venv\Scripts\deactivate.bat 2>nul
    )
    
    if !DEPS_INSTALLED! equ 0 (
        echo 正在安装依赖...
        echo.
        
        echo 请选择安装源:
        echo 1. 国内镜像源（推荐国内网络环境）
        echo 2. 官方源（推荐使用VPN或国外网络环境）
        echo.
        set /p source_choice="请输入选择 (1/2): "

        if "!source_choice!"=="1" (
            set PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
            set NPM_REGISTRY=https://registry.npmmirror.com
            echo 已选择: 国内镜像源
        ) else (
            set PIP_MIRROR=https://pypi.org/simple
            set NPM_REGISTRY=https://registry.npmjs.org
            echo 已选择: 官方源
        )
        echo.

        echo 正在清理可能存在的锁定文件...
        if exist "%APPDATA%\npm-cache\_locks" (
            del /f /s /q "%APPDATA%\npm-cache\_locks" > nul 2>&1
        )
        if exist "C:\Program Files\nodejs\node_cache\_locks" (
            del /f /s /q "C:\Program Files\nodejs\node_cache\_locks" > nul 2>&1
        )
        if exist "frontend\node_modules\.package-lock.json" (
            del /f /q "frontend\node_modules\.package-lock.json" > nul 2>&1
        )
        echo.

        echo 正在创建Python虚拟环境...
        if not exist venv (
            python -m venv venv || (
                echo 错误: 无法创建虚拟环境，请确保已安装Python
                goto error_pause
            )
        )
        call venv\Scripts\activate.bat || (
            echo 错误: 无法激活虚拟环境
            goto error_pause
        )
        echo.

        echo 正在安装后端依赖...
        echo 第1步: 升级pip...
        python -m pip install --upgrade pip -i !PIP_MIRROR!
        echo.

        echo 第2步: 安装Flask和基础依赖...
        python -m pip install flask==2.0.1 flask-cors==3.0.10 pillow==8.3.1 numpy==1.24.4 werkzeug==2.0.1 python-dotenv==0.19.0 -i !PIP_MIRROR!
        if !ERRORLEVEL! neq 0 (
            echo 警告: 部分基础依赖可能安装失败，继续安装...
        )
        echo.

        echo 第3步: 安装机器学习依赖...
        python -m pip install scikit-learn tqdm pandas -i !PIP_MIRROR!
        if !ERRORLEVEL! neq 0 (
            echo 警告: 部分机器学习依赖可能安装失败，继续安装...
        )
        echo.

        echo 第4步: 安装PyTorch...
        python -m pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html -i !PIP_MIRROR!
        if !ERRORLEVEL! neq 0 (
            echo 警告: PyTorch安装可能失败，继续安装...
        )
        echo.

        echo 第5步: 安装其他依赖...
        python -m pip install -r requirements.txt -i !PIP_MIRROR!
        if !ERRORLEVEL! neq 0 (
            echo 警告: 其他依赖安装可能不完整，继续安装...
        )
        echo.

        echo 第6步: 验证关键依赖...
        python -c "import flask; print('Flask版本:', flask.__version__)" 2>nul
        if !ERRORLEVEL! neq 0 (
            echo 错误: Flask安装失败，无法继续
            goto error_pause
        )
        echo Flask安装成功！
        echo.

        echo 正在配置npm镜像...
        cd frontend || (
            echo 错误: 无法进入frontend目录，请确保目录存在
            goto error_pause
        )
        call npm config set registry !NPM_REGISTRY!
        echo.

        echo 正在安装前端依赖...
        call npm install
        if !ERRORLEVEL! neq 0 (
            echo 错误: 前端依赖安装失败
            cd ..
            goto error_pause
        )
        echo 前端依赖安装完成！
        cd ..
        echo.
    )
)

:: 确保虚拟环境存在
if not exist "venv" (
    echo 错误: 未找到虚拟环境，请先安装依赖
    goto error_pause
)

:: 检查必要目录
if not exist "backend" (
    echo 错误: 未找到backend目录，请确保项目结构完整
    goto error_pause
)

if not exist "frontend" (
    echo 错误: 未找到frontend目录，请确保项目结构完整
    goto error_pause
)

:: 停止可能正在运行的服务
taskkill /F /IM "node.exe" 2>nul
taskkill /F /FI "WINDOWTITLE eq 后端服务" 2>nul

:: 启动后端服务
echo 正在启动后端服务...
start "后端服务" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && cd backend && python app.py"
if !ERRORLEVEL! neq 0 (
    echo 错误: 无法启动后端服务
    goto error_pause
)
echo 后端服务启动中，请稍候...
timeout /t 5 > nul

:: 启动前端服务
echo 正在启动前端服务...
cd frontend
start "前端服务" cmd /k "npm start"
cd ..
if !ERRORLEVEL! neq 0 (
    echo 错误: 无法启动前端服务
    goto error_pause
)

:: 等待服务启动
echo 等待服务启动...
timeout /t 5 > nul

echo.
echo ======================================================
echo 应用程序启动成功！
echo 前端地址: http://localhost:3000
echo 后端地址: http://localhost:5000
echo ======================================================
echo 关闭此窗口不会停止应用程序运行
echo 如需停止应用程序，请按Ctrl+C关闭命令行窗口
echo.
echo 按任意键退出此窗口...
pause > nul
exit /b 0

:error_pause
echo.
echo ======================================================
echo 启动过程中出现错误，请查看上面的错误信息
echo ======================================================
echo 按任意键退出...
pause > nul
exit /b 1 