@echo off
TITLE StyleTTS2
cd /d "%~dp0..\StyleTTS2"
call %USERPROFILE%\miniconda30\Scripts\activate.bat
call conda activate gem_mcp
call python watcher.py
cmd /k