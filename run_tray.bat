@echo off
REM Launches the Meridian system-tray app (green "M" icon) with no console window.
REM Double-click this file, or add a shortcut to it in your Startup folder to
REM launch the tray at Windows logon.
cd /d "%~dp0"
start "" /B "C:\Python314\pythonw.exe" "%~dp0meridian\tray\tray.py"
