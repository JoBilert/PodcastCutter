# PodcastCutter

Ein Python-Programm zur automatisierten Analyse von Audio-Dialogen mit Zeitcode-Transkription, LLM-basierter Themenanalyse und Benutzer-gesteuertem Audio-Schnitt.

Alle Verarbeitungsschritte laufen lokal. Es werden keine Online-Dienste benötigt.

## Features

*   **Lokale Audio-Transkription:** Wandelt eine MP3/WAV-Datei in ein strukturiertes Transkript mit Zeitstempeln um (nutzt `openai-whisper` lokal über `ffmpeg`, kein PyAV, keine Cloud-API nötig).
*   **Lokale LLM-Analyse:** Ein LLM über die **Unsloth-API** (vLLM-kompatibel, lokal laufend) sucht das Transkript nach Abschnitten, die thematisch zu einem vorgegebenen Thema passen (z.B. "Buddhismus und Mitgefühl").
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
*   `openai-whisper`, `openai`, `pydub`, `gradio` (siehe `requirements.txt`)
*   `ffmpeg` (muss systemweit installiert sein – wird von `openai-whisper` für Audio-Decoding benötigt)
*   Eine lokal laufende **Unsloth/vLLM API** (z. B. `http://127.0.0.1:8888/v1`) ODER OpenAI Cloud (Standard), konfigurierbar via GUI.

## Installation

```bash
# Projektverzeichnis betreten
cd /home/talmar/HermesProjekte/PodcastCutter

# Conda-Umgebung aktivieren (falls vorhanden)
conda activate podcastcutter

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Unsloth API lokal starten (Beispiel)

Unsloth stellt in der Regel einen vLLM-kompatiblen Server bereit. Starten Sie ihn z.B. so:

```bash
python -m unsloth.vllm.server --model unsloth/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8888
```

## Laufanweisung (GUI)

1.  Stellen Sie sicher, dass `ffmpeg` installiert ist und im PATH liegt.
2.  Starten Sie Ihre lokale Unsloth/vLLM API.
3.  Führen Sie die GUI aus:
    ```bash
    python3 app.py
    ```
    4.  Öffnen Sie die angezeigte URL (standardmäßig `http://localhost:7860`).
    5.  Tragen Sie Ihren **OpenAI API Key** ein (oder aktivieren Sie "Lokale Unsloth-API verwenden" und geben Sie die lokale Base URL an).
    6.  Laden Sie eine Podcast-Datei hoch, setzen Sie Thema und Modell, klicken Sie auf "Transkribieren & Analysieren".
6.  Prüfen Sie die markierten Passagen und klicken Sie auf "Schnitt bestätigen" oder geben Sie Feedback ein und klicken Sie auf "Erneut analysieren".
7.  Laden Sie den geschnittenen Podcast herunter.

## Abhängigkeiten (requirements.txt)

```
openai-whisper
httpx
pydub
gradio
```