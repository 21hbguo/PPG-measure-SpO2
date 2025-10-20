@echo off
echo Starting SpO2 Prediction Web App...
echo.

REM 切换到项目目录
C:
cd "C:\Users\26788\Desktop\pig_project\oximetry-phone-cam-data-main\whole"

REM 激活 Conda 环境并启动 Streamlit
conda activate py313
streamlit run app.py

echo Streamlit server stopped.
pause