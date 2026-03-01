TITLE Chat Browser
cd /d "%~dp0.."
:: Edit this line to match your installation path for Anaconda
call %USERPROFILE%\miniconda30\Scripts\activate.bat
call conda activate mcp_env_1
:: call Python mcp_v1b.py
call Python chat_browser.py
cmd /k