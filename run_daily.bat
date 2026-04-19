@echo off
REM ============================================================
REM 毎日のボートレースデータ取得 (Windows タスクスケジューラ用)
REM ------------------------------------------------------------
REM 使い方:
REM   1. タスクスケジューラで新しいタスクを作成
REM   2. トリガー: 毎日 00:15 (JST) — 前日の全レース終了後を想定
REM   3. 操作: このバッチを起動
REM      プログラム:  C:\Users\Sakan\OneDrive\Desktop\boatrace\boatrace-ai\run_daily.bat
REM      開始: C:\Users\Sakan\OneDrive\Desktop\boatrace\boatrace-ai
REM ------------------------------------------------------------
REM ログは logs\daily_YYYYMMDD_HHMMSS.log に残る
REM ============================================================

REM このバッチがあるディレクトリへ移動
cd /d %~dp0

REM タイムスタンプ (ログファイル名用)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set TS=%DT:~0,8%_%DT:~8,6%

REM venv があれば有効化 (なくても動くように)
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM 出力は UTF-8 で
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

REM ログ先フォルダ確保
if not exist logs mkdir logs

echo [%TS%] 開始 > logs\daily_%TS%.log 2>&1
python -m scripts.fetch_all --yesterday >> logs\daily_%TS%.log 2>&1
echo [%TS%] 終了 ExitCode=%ERRORLEVEL% >> logs\daily_%TS%.log 2>&1

REM 1週間より古いログを自動削除
forfiles /P logs /M daily_*.log /D -7 /C "cmd /c del @path" 2>nul

exit /b %ERRORLEVEL%
