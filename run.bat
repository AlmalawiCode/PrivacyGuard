@echo off
REM PrivacyGuard Run Script (Windows)
REM Author: Abdulmohsen Almalawi <balmalowy@kau.edu.sa>

echo ========================================
echo          PrivacyGuard Launcher
echo ========================================
echo.

REM Check if Ant is available
where ant >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Apache Ant is not installed or not in PATH.
    echo Please install Ant from: https://ant.apache.org/
    pause
    exit /b 1
)

REM Check if Java is available
where java >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Java is not installed or not in PATH.
    echo Please install Java JDK 18+
    pause
    exit /b 1
)

REM Compile and run
echo Building project...
call ant compile

if %errorlevel% equ 0 (
    echo.
    echo Starting PrivacyGuard...
    call ant run
) else (
    echo Build failed. Please check the error messages above.
    pause
    exit /b 1
)
