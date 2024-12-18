@echo off
set PREV_PYTHONPATH = %PYTHONPATH%
set PYTHONPATH=%cd%;%PYTHONPATH%
echo %PYTHONPATH%
python app/main.py %*
set PYTHONPATH=%PREV_PYTHONPATH%
endlocal