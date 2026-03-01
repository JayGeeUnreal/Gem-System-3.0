TITLE Vision
cd /d "%~dp0.."
call %USERPROFILE%\miniconda30\Scripts\activate.bat
call conda activate mcp_env_1
call Python vision.py
cmd /k