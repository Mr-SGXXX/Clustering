@REM MIT License

@REM Copyright (c) 2023-2024 Yuxuan Shao

@REM Permission is hereby granted, free of charge, to any person obtaining a copy
@REM of this software and associated documentation files (the "Software"), to deal
@REM in the Software without restriction, including without limitation the rights
@REM to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
@REM copies of the Software, and to permit persons to whom the Software is
@REM furnished to do so, subject to the following conditions:

@REM The above copyright notice and this permission notice shall be included in all
@REM copies or substantial portions of the Software.

@REM THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
@REM IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
@REM FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
@REM AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
@REM LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
@REM OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
@REM SOFTWARE.





@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM Check if the correct number of arguments is provided
IF "%~5" NEQ "" (
    echo Usage: %0 conda_env_name run_count [config_file] [device]
    exit /b 1
)

REM Assign arguments to variables
SET conda_env_name=%1
SET run_count=%2

IF "%~3" NEQ "" (
    SET config_file=%3
) ELSE (
    SET config_file=cfg\example.cfg
)

IF "%~4" NEQ "" (
    SET device=%4
)

REM Activate the specified conda environment
CALL conda activate %conda_env_name%

REM Navigate to the directory containing main.py
cd ..

REM Copy the config file in case it is changed during the run
REM Get the current date and time
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value ^| find "="') do set datetime=%%i
set "timestamp=%datetime:~0,8%_%datetime:~8,6%"
set "unique_name=config_%timestamp%.cfg"

REM Create the .temp directory if it doesn't exist
if not exist .temp (
    mkdir .temp
)

copy "%config_file%" ".temp\%unique_name%"
set "config_file=.temp\%unique_name%"

REM Run the Python script the specified number of times
FOR /L %%i IN (1,1,%run_count%) DO (
    echo Running iteration %%i
    IF DEFINED device (
        python main.py -cp "!config_file!" -d "!device!"
    ) ELSE (
        python main.py -cp "!config_file!"
    )
)

REM Remove the copied config file
del %config_file%

REM Deactivate the conda environment
CALL conda deactivate

REM Navigate back to the scripts directory
cd scripts

ENDLOCAL
