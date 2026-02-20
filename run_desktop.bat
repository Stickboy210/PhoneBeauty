@echo off
chcp 65001 >nul
echo ==========================================
echo          美颜相机 - 桌面版
echo ==========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo [1/3] 检查Python版本...
python --version
echo.

REM 检查虚拟环境
if not exist "venv" (
    echo [2/3] 创建虚拟环境...
    python -m venv venv
) else (
    echo [2/3] 虚拟环境已存在
)

echo.
echo [3/3] 安装/更新依赖...
call venv\Scripts\activate
pip install -q -r requirements.txt

echo.
echo ==========================================
echo 启动美颜相机...
echo ==========================================
python main.py

echo.
echo 应用已退出
pause
