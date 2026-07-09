@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cd /d "d:\a_task1\Thesis"
set PYTHONUTF8=1
python deploy\compile_hmst_cuda.py
exit /b %ERRORLEVEL%
