@echo off
TITLE Chat Logger
cd /d "%~dp0..\top_5"
:: Edit this line to match your installation path for Anaconda
call %USERPROFILE%\miniconda30\Scripts\activate.bat
::
call conda activate mcp_env_1
call python ssn_chat_saver.py
cmd /k