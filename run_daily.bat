@echo off
REM ============================================================
REM 毎日のボートレースデータ取得 (Windows タスクスケジューラ用)
REM ------------------------------------------------------------
REM   scripts\run_daily.py がギャップ検出つきで fetch_all を呼ぶ。
REM   PCがN日間起動していなくても、次回起動時にまとめて補完される。
REM
REM タスクスケジューラ登録例:
REM   プログラム: C:\Users\Sakan\OneDrive\Desktop\boatrace\boatrace-ai\run_daily.bat
REM   開始:        C:\Users\Sakan\OneDrive\Desktop\boatrace\boatrace-ai
REM   トリガー:    毎日 00:30 JST
REM ============================================================

cd /d %~dp0

if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

if not exist logs mkdir logs

python -m scripts.run_daily
set RC=%ERRORLEVEL%

REM 30日より古いログを自動削除 (容量節約)
forfiles /P logs /M run_daily_*.log /D -30 /C "cmd /c del @path" 2>nul
forfiles /P logs /M fetch_*.log /D -30 /C "cmd /c del @path" 2>nul

exit /b %RC%
