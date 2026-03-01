TITLE Control Panel
cd /d "%~dp0.."
call %USERPROFILE%\miniconda30\Scripts\activate.bat
call conda activate mcp_env_1
python control_panel.py
cmd /k