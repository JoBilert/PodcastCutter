@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ==================================================
echo  PodcastCutter - Selbstkontainiertes Startskript
echo ==================================================
echo.

REM ---------------------------------------------------------------------
REM 0. Projektverzeichnis
REM ---------------------------------------------------------------------
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"
if not exist "app.py" (
    echo FEHLER: app.py nicht gefunden.
    echo Bitte starten Sie dieses Skript aus dem PodcastCutter-Verzeichnis.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------
REM 1. Python prüfen und ggf. installieren
REM ---------------------------------------------------------------------
echo [1/6] Prüfe Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python nicht gefunden. Installation via winget...
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements >nul 2>&1
    REM Winget aktualisiert PATH nicht in der aktuellen Sitzung.
    REM Sucht python.exe in typischen Installationspfaden und fügt zum PATH hinzu.
    for %%v in (3.12 3.11 3.10) do (
        if exist "%LocalAppData%\Programs\Python\Python%%v\python.exe" (
            set PATH=%PATH%;%LocalAppData%\Programs\Python\Python%%v\;%LocalAppData%\Programs\Python\Python%%v\Scripts\
        )
    )
    python --version >nul 2>&1
    if errorlevel 1 (
        echo FEHLER: Python konnte nicht installiert werden oder ist nicht im PATH.
        echo Bitte installieren Sie Python 3.11 oder 3.12 manuell von https://python.org
        echo und starten Sie dieses Skript erneut.
        pause
        exit /b 1
    )
    echo Python installiert und PATH aktualisiert.
) else (
    for /f "tokens=2 delims=. " %%v in ('python --version 2^>nul') do set PYTHON_VERSION=%%v
    echo Python gefunden: %PYTHON_VERSION%
)

REM ---------------------------------------------------------------------
REM 2. ffmpeg prüfen und ggf. installieren
REM ---------------------------------------------------------------------
echo [2/6] Prüfe ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ffmpeg nicht gefunden. Installation via winget...
    winget install ffmpeg --accept-package-agreements --accept-source-agreements >nul 2>&1
    REM Sucht ffmpeg.exe in typischen winget-Installationspfaden
    for /f "delims=" %%f in ('dir /s /b "%LocalAppData%\Microsoft\WinGet\Packages\ffmpeg*\ffmpeg.exe" 2^>nul') do set "FFMPEG_DIR=%%~dpf"
    if defined FFMPEG_DIR set PATH=%PATH%;%FFMPEG_DIR%
    ffmpeg -version >nul 2>&1
    if errorlevel 1 (
        echo FEHLER: ffmpeg konnte nicht installiert werden.
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
REM 3. Virtuelle Umgebung erstellen (nur wenn nicht vorhanden)
REM ---------------------------------------------------------------------
echo [3/6] Prüfe virtuelle Umgebung...
if not exist ".venv\Scripts\activate.bat" (
    echo Erstelle virtuelle Umgebung (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo FEHLER: Konnte virtuelle Umgebung nicht erstellen.
        pause
        exit /b 1
    )
    echo Virtuelle Umgebung erstellt.
) else (
    echo Virtuelle Umgebung vorhanden.
)

REM ---------------------------------------------------------------------
REM 4. Abhängigkeiten installieren (nur beim ersten Mal)
REM ---------------------------------------------------------------------
echo [4/6] Prüfe Python-Abhängigkeiten...
call .venv\Scripts\activate.bat
if not exist ".venv\installed.ok" (
    echo Installiere Abhängigkeiten...
    pip install --upgrade pip >nul 2>&1
    pip install -r requirements.txt
    if errorlevel 1 (
        echo FEHLER: Konnte Abhängigkeiten nicht installieren.
        pause
        exit /b 1
    )
    echo . > ".venv\installed.ok"
    echo Abhängigkeiten installiert.
) else (
    echo Abhängigkeiten bereits installiert.
)

REM ---------------------------------------------------------------------
REM 5. App starten und warten bis Server bereit ist
REM ---------------------------------------------------------------------
echo [5/6] Starte PodcastCutter GUI...
echo Die Anwendung lädt unter: http://localhost:7860
echo Drücken Sie STRG+C zum Beenden des Servers.
echo.

REM Server im Hintergrund starten
start "" python app.py

REM Warten bis der Server antwortet (max 60 Sekunden)
echo Wartet auf Server-Start...
set /a count=0
:waitloop
curl -s http://localhost:7860 >nul 2>&1
if errorlevel 1 (
    set /a count+=1
    if %count% geq 120 (
        echo Warnung: Server nicht bereit nach 60 Sekunden.
        echo Öffne Browser trotzdem.
        start "" "microsoft-edge:http://localhost:7860" >nul 2>&1
        goto :started
    )
    timeout /t 1 >nul
    goto :waitloop
)
echo Server bereit.

:started
REM Browser öffnen
echo Öffne Edge...
start "" "microsoft-edge:http://localhost:7860" >nul 2>&1

REM ---------------------------------------------------------------------
REM 6. Aufräumen beim Beenden
REM ---------------------------------------------------------------------
:end
echo.
echo PodcastCutter wurde beendet.
deactivate >nul 2>&1
endlocal