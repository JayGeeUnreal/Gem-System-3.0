@echo off
TITLE Watcher To Face
cd /d "%~dp0..\Neurosync\NeuroSync_Player"
call %USERPROFILE%\miniconda30\Scripts\activate.bat
call conda activate mcp_env_1
python watcher_to_face.py
cmd /k