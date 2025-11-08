@echo off
:: 设置 Anaconda/Miniconda 的安装路径（根据你的实际路径修改）
call D:\Users\31649\anaconda3\Scripts\activate.bat yolov8

:: 检查环境是否激活成功
if %ERRORLEVEL% NEQ 0 (
    echo 环境激活失败，请检查环境名或Conda路径。
    pause
    exit /b %ERRORLEVEL%
)

:: 运行 Python 脚本
python gui_detector.py

:: 可选：脚本结束后暂停，查看输出结果
pause