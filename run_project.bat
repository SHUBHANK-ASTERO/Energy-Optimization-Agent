@echo off
echo Installing requirements...
python -m pip install -r requirements.txt
echo.
echo Starting Energy Optimization Agent...
python app.py
pause



cd D:\Energy_Optimization_Agent_Complete\Energy_Optimization_Agent_Complete
python --version
python -m pip install -r requirements.txt
python app.py