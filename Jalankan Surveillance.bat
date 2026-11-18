@echo off
rem Menjalankan server deteksi AI lalu membuka dashboard di browser.
cd /d "%~dp0app"
if not exist "models\helmet_best.pt" py -3 download_models.py
start "" http://localhost:8000
py -3 server.py
pause
