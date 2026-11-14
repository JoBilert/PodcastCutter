# PodcastCutter

Ein Python-Programm zur automatisierten Analyse von Audio-Dialogen mit Zeitcode-Transkription, LLM-basierter Themenanalyse und Benutzer-gesteuertem Audio-Schnitt.

Alle Verarbeitungsschritte laufen lokal (Transkription via openai-whisper).
Die LLM-Analyse erfolgt über eine beliebige OpenAI-kompatible API (NVIDIA NIM, OpenAI,
OpenRouter, lokale Unsloth/vLLM oder eigene API).

## Schnellstart (Windows)

1. Laden Sie dieses Repository herunter (oder klonen Sie es).
2. Doppelklicken Sie auf **`start.bat`**.
3. Das Skript erledigt alles Weitere:
   * Prüft und installiert ggf. Python 3.12
   * Prüft und installiert ggf. ffmpeg
   * Erstellt eine virtuelle Python-Umgebung (`.venv`)
   * Installiert alle Python-Abhängigkeiten (`openai-whisper`, `pydub`, `gradio`)
   * Startet die GUI
   * Öffnet Microsoft Edge unter `http://localhost:7860`

## Voraussetzungen

*   **Windows 10 oder 11** (für `start.bat` und winget)
*   **Administratorrechte** (falls Python oder ffmpeg nachinstalliert werden müssen)
*   **Internetverbindung** (für den ersten Start zum Installieren von Python/ffmpeg/Paketen)
*   **NVIDIA NIM API Key** (oder OpenAI/OpenRouter/Unsloth) – bei erstem Start in Tab 0 eingeben

## Was passiert beim ersten Start?

1. `start.bat` prüft, ob Python und ffmpeg installiert sind. Falls nicht, werden sie automatisch über `winget` installiert.
2. Eine virtuelle Umgebung unter `.venv` wird erstellt.
3. `pip install -r requirements.txt` installiert alle Python-Pakete.
4. Die Gradio-GUI startet.
5. Microsoft Edge öffnet sich automatisch mit `http://localhost:7860`.

## API-Einrichtung (Tab 0)

Beim ersten Start der GUI öffnet sich automatisch der Tab **„0. 🔧 Einrichtung“**:

1. Wählen Sie einen API-Anbieter (NVIDIA NIM, OpenAI, OpenRouter, Unsloth/vLLM oder eigene API).
2. Tragen Sie API Key, Base URL und Modellname ein.
3. Klicken Sie auf „Konfiguration speichern“.
4. Wechseln Sie zu Tab 1 und laden Sie Ihre Podcast-Datei hoch.

### NVIDIA NIM Zugang

1. Gehen Sie auf https://build.nvidia.com/nim und erstellen Sie ein Konto.
2. Suchen Sie nach einem LLM-Modell (z. B. `nemotron-3.5-super-vl-120b-a12b`).
3. Erstellen Sie einen API-Key unter "API Keys" im Dashboard.

### Lokale Unsloth/vLLM

1. Starten Sie einen lokalen Server, z. B.:
   ```bash
   python -m unsloth.vllm.server --model unsloth/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8888
   ```
2. Base URL: `http://127.0.0.1:8888/v1`
3. API-Key kann leer bleiben oder "not-needed" sein.

## Manuelle Installation (ohne start.bat)

```bash
# Projektverzeichnis betreten
cd /home/talmar/HermesProjekte/PodcastCutter

# Abhängigkeiten installieren
pip install -r requirements.txt

# GUI starten
python3 app.py
```

## Projektstruktur

```
PodcastCutter/
├── app.py               # Gradio-GUI (Haupt-Einstieg: python3 app.py)
├── audio_processor.py   # CLI-Workflow (alternativ ohne GUI)
├── start.bat            # Selbstkontainierter Windows-Start (Python, ffmpeg, venv, Edge)
├── PLAN.md              # Projektplan und nächste Schritte
├── README.md            # Diese Dokumentation
├── requirements.txt     # Python-Abhängigkeiten
├── input_dialog.mp3     # Platzhalter-Audio (muss vom Nutzer bereitgestellt werden)
├── .gitignore           # Git-Ausschlussregeln
└── .env                 # Konfiguration (wird bei erstem Lauf erstellt, NICHT ins Repository commiten)
```

## Git-Branches

*   `master` – Lokale Arbeitsversion (kann `.env` mit API-Key enthalten)
*   `clean-setup` – Saubere Version ohne API-Geheimnisse, mit erstmaligem Setup-Flow. `.env` ist im `.gitignore` und wird nicht committet.

Branch wechseln:
```bash
git checkout clean-setup   # Saubere Version ohne Secrets
git checkout master        # Lokale Version
```

## Credits & Lizenzen

Dieses Projekt nutzt folgende Bibliotheken und Systeme:

| Komponente | Lizenz | Quelle |
|------------|--------|--------|
| Python | [Python Software Foundation License](https://docs.python.org/3/license.html) | https://python.org |
| openai-whisper | [MIT License](https://github.com/openai/whisper/blob/main/LICENSE) | https://github.com/openai/whisper |
| pydub | [MIT License](https://github.com/jiaaro/pydub/blob/master/LICENSE) | https://github.com/jiaaro/pydub |
| gradio | [Apache License 2.0](https://github.com/gradio-app/gradio/blob/main/LICENSE) | https://github.com/gradio-app/gradio |
| ffmpeg | [GNU General Public License v2+](https://ffmpeg.org/legal.html) | https://ffmpeg.org |
| NVIDIA NIM API | NVIDIA Terms of Service (proprietär) | https://build.nvidia.com/nim |
| OpenAI API | OpenAI Terms of Service (proprietär) | https://openai.com/policies/terms-of-use |
| OpenRouter API | OpenRouter Terms of Service (proprietär) | https://openrouter.ai |

Dieses Projekt steht unter der MIT License. Siehe [LICENSE](LICENSE) für Details (falls vorhanden).

**Hinweis:** Die Nutzung von Cloud-APIs (NVIDIA NIM, OpenAI, OpenRouter) unterliegt den jeweiligen Nutzungsbedingungen des Anbieters und kann kostenpflichtig sein. Die lokale Transkription via `openai-whisper` läuft vollständig offline auf Ihrer eigenen Hardware.