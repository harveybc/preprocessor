@echo off
setlocal
set PREV_PYTHONPATH = %PYTHONPATH%
set PYTHONPATH= "$PWD";%PYTHONPATH%
echo %PYTHONPATH%
python app/main.py %*
set PYTHONPATH=%PREV_PYTHONPATH%
endlocal