@echo off
chcp 65001 > nul
echo ======================================================
echo 植物病害识别系统 - 停止服务
echo ======================================================
echo.

echo 正在停止所有相关服务...

echo 停止前端服务 (端口3000)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo 前端服务已停止
    )
)

echo 停止后端服务 (端口5000)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo 后端服务已停止
    )
)

echo.
echo ======================================================
echo 所有服务已停止！
echo ======================================================
pause 