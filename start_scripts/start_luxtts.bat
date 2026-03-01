@echo off
TITLE LuxTTS
cd /d "%~dp0..\LuxTTS"
:: Edit this line to match your installation path for Anaconda
call %USERPROFILE%\miniconda30\Scripts\activate.bat
::
call conda activate mcp_env_1
call python luxtts_server.py
cmd /k