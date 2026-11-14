# PodcastCutter

Ein Python-Programm zur automatisierten Analyse von Audio-Dialogen mit Zeitcode-Transkription, LLM-basierter Themenanalyse und Benutzer-gesteuertem Audio-Schnitt.

Alle Verarbeitungsschritte laufen lokal (Transkription via openai-whisper).
Die LLM-Analyse erfolgt über eine beliebige OpenAI-kompatible API (NVIDIA NIM, OpenAI,
OpenRouter, lokale Unsloth/vLLM oder eigene API).

## Features

*   **Lokale Audio-Transkription:** Wandelt eine MP3/WAV-Datei in ein strukturiertes Transkript mit Zeitstempeln um (nutzt `openai-whisper` lokal über `ffmpeg`, kein PyAV, keine Cloud-API nötig).
*   **Beliebige LLM-APIs:** Unterstützt NVIDIA NIM, OpenAI, OpenRouter, lokale Unsloth/vLLM oder jede eigene OpenAI-kompatible API. Auswahl über Dropdown in der GUI.
*   **Gradio-GUI:** Bequeme Oberfläche zum Hochladen von Podcast-Dateien, Einstellen von Thema/Modell/URL, Anzeige von Transkript und markierten Passagen sowie Bestätigung des Schnitts.
*   **Bestätigungs-Schleife:** Bei Ablehnung wird Feedback eingeholt und der Such/Markierungsvorgang wiederholt.
*   **Audio-Schnitt:** Schneidet den Original-Audio-Track exakt auf die Start- und Endpunkte der bestätigten Passagen zu und speichert das Ergebnis als `output_trimmed.mp3`.

## Projektstruktur

```
PodcastCutter/
├── app.py               # Gradio-GUI (Haupt-Einstieg: python3 app.py)
├── audio_processor.py   # CLI-Workflow (alternativ ohne GUI)
├── PLAN.md              # Projektplan und nächste Schritte
├── README.md            # Diese Dokumentation
├── requirements.txt     # Python-Abhängigkeiten
├── input_dialog.mp3     # Platzhalter-Audio (muss vom Nutzer bereitgestellt werden)
└── output_trimmed.mp3   # Ausgabe-Audio (wird vom Programm generiert)
```

## Voraussetzungen

*   Python 3.x (empfohlen: 3.11 oder 3.12)
*   `openai-whisper`, `pydub`, `gradio` (siehe `requirements.txt`)
*   `ffmpeg` (muss systemweit installiert sein – wird von `openai-whisper` für Audio-Decoding benötigt)
*   Eine API für die LLM-Analyse (NVIDIA NIM, OpenAI, OpenRouter, lokale Unsloth/vLLM oder eigene), konfigurierbar via GUI.

## Installation

```bash
# Projektverzeichnis betreten
cd /home/talmar/HermesProjekte/PodcastCutter

# Conda-Umgebung aktivieren (falls vorhanden)
conda activate podcastcutter

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Erstlauf – Konfiguration

Beim ersten Start öffnet sich automatisch Tab **„0. 🔧 Einrichtung“**:

1. Wählen Sie einen API-Anbieter aus dem Dropdown (NVIDIA NIM, OpenAI, OpenRouter, Unsloth/vLLM oder eigene API).
2. Tragen Sie API Key, Base URL und Modellname ein.
3. Legen Sie Ausgabe-Pfade für das geschnittene MP3 und den Transkript-Cache fest.
4. Klicken Sie auf „Konfiguration speichern“. Die Einstellungen werden in einer `.env`-Datei gespeichert (chmod 600, nur für Sie lesbar).
5. Wechseln Sie zu Tab 1 und nutzen Sie das Programm.

### NVIDIA NIM Zugang

1. Gehen Sie auf https://build.nvidia.com/nim und erstellen Sie ein Konto.
2. Suchen Sie nach einem LLM-Modell (z. B. `nemotron-3.5-super-vl-120b-a12b`).
3. Erstellen Sie einen API-Key unter "API Keys" im Dashboard.
4. Tragen Sie Key, Base URL (`https://integrate.api.nvidia.com/v1`) und Modellname ein.

### Lokale Unsloth/vLLM

1. Starten Sie einen lokalen Server, z. B.:
   ```bash
   python -m unsloth.vllm.server --model unsloth/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8888
   ```
2. Base URL: `http://127.0.0.1:8888/v1`
3. Modellname: wie beim Serverstart angegeben.
4. API-Key kann leer bleiben oder "not-needed" sein.

## Laufanweisung (GUI)

1.  Stellen Sie sicher, dass `ffmpeg` installiert ist und im PATH liegt.
2.  Führen Sie die GUI aus:
    ```bash
    python3 app.py
    ```
3.  Öffnen Sie die angezeigte URL (standardmäßig `http://localhost:7860`).
4.  Falls noch nicht geschehen: Richten Sie in Tab 0 Ihre API-Zugangsdaten ein.
5.  Laden Sie eine Podcast-Datei hoch, setzen Sie Thema und Modell, klicken Sie auf "Transkribieren & Analysieren".
6.  Prüfen Sie die markierten Passagen und klicken Sie auf "Schnitt bestätigen" oder geben Sie Feedback ein und klicken Sie auf "Erneut analysieren".
7.  Laden Sie den geschnittenen Podcast herunter.

## Abhängigkeiten (requirements.txt)

```
openai-whisper
pydub
gradio
```

## Git-Branches

*   `master` – Lokale Arbeitsversion (kann `.env` mit API-Key enthalten)
*   `clean-setup` – Saubere Version ohne API-Geheimnisse, mit erstmaligem Setup-Flow. `.env` ist im `.gitignore` und wird nicht committet.

Branch wechseln:
```bash
git checkout clean-setup   # Saubere Version ohne Secrets
git checkout master        # Lokale Version
```