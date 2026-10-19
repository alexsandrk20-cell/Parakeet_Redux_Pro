@echo off
title Parakeet Redux Local Transcriber
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py --server.maxUploadSize 10240
pause
