@echo off
rem Double-click to start the eye tracking test. The first time it installs uv, then Python and the
rem app's packages (a few minutes, needs internet); after that it starts in seconds.
cd /d "%~dp0"
where uv >nul 2>nul || powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv run main.py
if errorlevel 1 pause
