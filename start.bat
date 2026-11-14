@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ==================================================
echo  PodcastCutter - Selbstkontainiertes Startskript
echo ==================================================
echo.

REM ---------------------------------------------------------------------
REM 1. Prüfen ob im Projektverzeichnis ausgeführt wird
REM ---------------------------------------------------------------------
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"
if not exist "app.py" (
    echo FEHLER: app.py nicht gefunden. Bitte starten Sie dieses Skript
    echo aus dem PodcastCutter-Projektverzeichnis.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------
REM 2. Python prüfen und ggf. installieren (via winget)
REM ---------------------------------------------------------------------
echo [1/5] Prüfe Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python nicht gefunden. Versuche Installation via winget...
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements >nul 2>&1
    if errorlevel 1 (
        echo FEHLER: Python konnte nicht automatisch installiert werden.
        echo Bitte installieren Sie Python 3.11 oder 3.12 manuell von https://python.org
        echo und starten Sie dieses Skript erneut.
        pause
        exit /b 1
    )
    echo Python installiert.
) else (
    echo Python gefunden: %PYTHON_VERSION%
)

REM ---------------------------------------------------------------------
REM 3. ffmpeg prüfen und ggf. installieren (via winget)
REM ---------------------------------------------------------------------
echo [2/5] Prüfe ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ffmpeg nicht gefunden. Versuche Installation via winget...
    winget install ffmpeg --accept-package-agreements --accept-source-agreements >nul 2>&1
    if errorlevel 1 (
        echo FEHLER: ffmpeg konnte nicht automatisch installiert werden.
        echo Bitte installieren Sie ffmpeg manuell von https://ffmpeg.org/download.html
        echo und stellen Sie sicher, dass es im PATH ist.
        pause
        exit /b 1
    )
    echo ffmpeg installiert.
) else (
    echo ffmpeg gefunden.
)

REM ---------------------------------------------------------------------
REM 4. Virtuelle Umgebung erstellen und Abhängigkeiten installieren
REM ---------------------------------------------------------------------
echo [3/5] Erstelle virtuelle Umgebung (.venv)...
python -m venv .venv >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Konnte virtuelle Umgebung nicht erstellen.
    pause
    exit /b 1
)

echo [4/5] Installiere Python-Abhängigkeiten...
call .venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo FEHLER: Konnte Abhängigkeiten nicht installieren.
    pause
    exit /b 1
)
echo Abhängigkeiten installiert.

REM ---------------------------------------------------------------------
REM 5. App starten und Edge öffnen
REM ---------------------------------------------------------------------
echo [5/5] Starte PodcastCutter GUI...
echo.
echo Die Anwendung läuft unter: http://localhost:7860
echo Öffnen Sie Edge falls das Fenster nicht automatisch öffnet.
echo.
echo Drücken Sie STRG+C zum Beenden des Servers.
echo.

REM Server im Hintergrund starten
start "" "http://localhost:7860" /d "%SCRIPT_DIR%" "microsoft-edge:http://localhost:7860" >nul 2>&1

call python app.py

REM Aufräumen beim Beenden
deactivate >nul 2>&1
endlocal