@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

:: 设置错误处理
set "ERROR_FLAG=0"

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
echo 植物病害智能识别系统 - 一键启动脚本 (新版)
echo ======================================================
echo.

echo 请选择操作:
echo 1. 首次安装依赖并启动应用 (新环境)
echo 2. 仅启动应用 (依赖已安装)
echo 3. 使用conda环境安装依赖并启动 (推荐)
echo.
set /p op_choice="请输入选择 (1/2/3): "

:: 检查输入是否有效
if not "%op_choice%"=="1" if not "%op_choice%"=="2" if not "%op_choice%"=="3" (
    echo 错误: 无效的选择，请输入1、2或3
    goto error_pause
)

:: 选择3: 使用conda环境
if "%op_choice%"=="3" (
    echo.
    echo 正在检查Conda是否可用...
    where conda >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo 错误: 未找到conda命令。请确保已安装Anaconda或Miniconda。
        goto error_pause
    )
    
    echo.
    echo 清除代理设置...
    set HTTP_PROXY=
    set HTTPS_PROXY=
    set ALL_PROXY=
    echo 代理设置已清除
    
    echo.
    echo 正在创建conda环境 plant_disease...
    call conda create -n plant_disease python=3.9 -y
    if !ERRORLEVEL! neq 0 (
        echo 警告: conda环境创建失败，尝试使用已有环境...
    )
    
    echo.
    echo 激活conda环境...
    call conda activate plant_disease
    if !ERRORLEVEL! neq 0 (
        echo 错误: 无法激活conda环境
        goto error_pause
    )
    
    echo.
    echo 安装Python依赖...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if !ERRORLEVEL! neq 0 (
        echo 警告: 部分Python依赖可能安装失败，继续安装...
        set "ERROR_FLAG=1"
    )
    
    echo.
    echo 进入前端目录并安装依赖...
    cd frontend
    call npm config set registry https://registry.npmmirror.com
    echo 安装前端依赖...
    call npm install
    if !ERRORLEVEL! neq 0 (
        echo 警告: 前端依赖安装可能不完整
        set "ERROR_FLAG=1"
    )
    cd ..
    
    goto start_services
)

:: 选择1或2: 使用Python venv环境
if "%op_choice%"=="1" (
    echo.
    echo 清除代理设置...
    set HTTP_PROXY=
    set HTTPS_PROXY=
    set ALL_PROXY=
    echo 代理设置已清除
    
    echo.
    echo 检查Python是否可用...
    where python >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo 错误: 未找到python命令。请确保已安装Python 3.9+。
        goto error_pause
    )
    
    echo.
    echo 创建Python虚拟环境...
    if not exist venv (
        python -m venv venv
        if !ERRORLEVEL! neq 0 (
            echo 错误: 无法创建虚拟环境
            goto error_pause
        )
    )
    
    echo.
    echo 激活虚拟环境...
    call venv\Scripts\activate.bat
    if !ERRORLEVEL! neq 0 (
        echo 错误: 无法激活虚拟环境
        goto error_pause
    )
    
    echo.
    echo 安装Python依赖...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if !ERRORLEVEL! neq 0 (
        echo 警告: 部分Python依赖可能安装失败，继续安装...
        set "ERROR_FLAG=1"
    )
    
    echo.
    echo 进入前端目录并安装依赖...
    cd frontend
    call npm config set registry https://registry.npmmirror.com
    echo 安装前端依赖...
    call npm install
    if !ERRORLEVEL! neq 0 (
        echo 警告: 前端依赖安装可能不完整
        set "ERROR_FLAG=1"
    )
    cd ..
) else if "%op_choice%"=="2" (
    echo.
    echo 检查虚拟环境是否存在...
    if not exist venv (
        echo 错误: 未找到虚拟环境，请先安装依赖
        goto error_pause
    )
    
    echo.
    echo 激活虚拟环境...
    call venv\Scripts\activate.bat
    if !ERRORLEVEL! neq 0 (
        echo 错误: 无法激活虚拟环境
        goto error_pause
    )
)

:start_services
:: 检查模型文件
echo.
echo 检查模型文件...
if not exist "models\best_model.pth" (
    echo 警告: 未找到模型文件 (models\best_model.pth)
    echo 请确保已下载模型文件并放置在正确位置
    echo 系统将继续启动，但识别功能可能无法正常工作
    set "ERROR_FLAG=1"
)

:: 停止可能正在运行的相关进程
echo.
echo 停止可能正在运行的服务...
taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq 后端服务*" >nul 2>&1
taskkill /F /IM "node.exe" /FI "WINDOWTITLE eq 前端服务*" >nul 2>&1

:: 设置Node.js OpenSSL Legacy Provider
echo.
echo 设置Node.js兼容性选项...
set NODE_OPTIONS=--openssl-legacy-provider

:: 启动后端服务
echo.
echo 启动后端服务...
start "后端服务" cmd /k "cd /d %~dp0 && cd backend && python app.py"
if !ERRORLEVEL! neq 0 (
    echo 错误: 无法启动后端服务
    goto error_pause
)
echo 后端服务启动中，请稍候...
timeout /t 5 > nul

:: 启动前端服务
echo.
echo 启动前端服务...
start "前端服务" cmd /k "cd /d %~dp0 && cd frontend && set NODE_OPTIONS=--openssl-legacy-provider && npm start"
if !ERRORLEVEL! neq 0 (
    echo 错误: 无法启动前端服务
    goto error_pause
)

:: 等待服务启动
echo 等待服务启动...
timeout /t 5 > nul

echo.
echo ======================================================
if "!ERROR_FLAG!"=="1" (
    echo 应用程序启动完成，但存在警告！
    echo 某些功能可能无法正常工作，请查看上面的警告信息。
) else (
    echo 应用程序启动成功！
)
echo 前端地址: http://localhost:3000
echo 后端地址: http://localhost:5000
echo ======================================================
echo 关闭此窗口不会停止应用程序运行
echo 如需停止应用程序，请关闭前端和后端的命令行窗口
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