# app.py
"""
PodcastCutter GUI — Gradio-Interface für lokale Audio-Transkription,
LLM-basierte Themenanalyse über beliebige OpenAI-kompatible APIs und
Benutzer-gesteuerten Audio-Schnitt.

Alle Verarbeitungsschritte laufen lokal (Transkription via openai-whisper).
Die LLM-Analyse erfolgt über eine konfigurierbare API (NVIDIA NIM, OpenAI,
OpenRouter, lokale Unsloth/vLLM oder eigene API).

Start:
    python3 app.py

Erstlauf:
    Richten Sie in Tab "0. Einrichtung" Ihre API-Zugangsdaten ein.
"""

import os
import json
import time
import gradio as gr
from pydub import AudioSegment
from datetime import timedelta
import whisper
import urllib.request

# ---------------------------------------------------------------------------
# Übersetzungsystem (DE / EN)
# ---------------------------------------------------------------------------

TRANSLATIONS = {
    "de": {
        # App
        "app_title": "✂️ PodcastCutter",
        "app_subtitle": "Lokale Audio-Transkription, LLM-Themenanalyse und benutzergesteuerter Schnitt. Transkription läuft lokal via openai-whisper; LLM-Analyse über eine konfigurierbare API.",
        "language_label": "Sprache / Language",
        # Tabs
        "tab_setup": "0. 🔧 Einrichtung",
        "tab_setup_done": "0. ✅ Einrichtung",
        "tab_settings": "1. Einstellungen & Datei",
        "tab_results": "2. Ergebnisse & Bestätigung",
        "tab_cut": "3. Schnitt & Download",
        # Setup Tab
        "setup_title": "## Erste Einrichtung",
        "setup_intro": "### API-Anbieter auswählen und Zugangsdaten eingeben\n\nWählen Sie den LLM-Anbieter aus, den Sie für die Themenanalyse verwenden möchten. Unterstützt werden alle OpenAI-kompatiblen APIs.",
        "setup_provider_label": "API-Anbieter",
        "setup_provider_info": "Welche LLM-API möchten Sie verwenden?",
        "setup_api_key_label": "API Key",
        "setup_api_key_placeholder": "nvapi-... oder sk-...",
        "setup_base_url_label": "Base URL",
        "setup_base_url_placeholder": "https://integrate.api.nvidia.com/v1",
        "setup_model_label": "LLM-Modell",
        "setup_model_placeholder": "z. B. nvidia/nemotron-3.5-super-vl-120b-a12b",
        "setup_output_audio_label": "Ausgabe-Datei für geschnittenes MP3",
        "setup_output_audio_placeholder": "/pfad/zum/ausgabe.mp3",
        "setup_output_transcript_label": "Ausgabe-Datei für Transkript-Cache",
        "setup_output_transcript_placeholder": "/pfad/zum/transkript.json",
        "setup_save_btn": "💾 Konfiguration speichern",
        "setup_save_success": "✅ Konfiguration gespeichert in .env\nSie können jetzt zu Tab '1. Einstellungen & Datei' wechseln und das Programm nutzen.",
        "setup_save_error_missing": "❌ Bitte füllen Sie alle Pflichtfelder aus (API Key, Base URL, Modell).",
        "setup_already_configured": "Ihre Konfiguration ist bereits gespeichert (`.env`). Sie können das Programm jetzt nutzen.\nFalls Sie die Einstellungen ändern möchten, bearbeiten Sie die `.env`-Datei direkt oder löschen Sie sie und starten Sie das Programm neu.",
        "setup_show_cfg_btn": "📋 Konfiguration anzeigen",
        "setup_cfg_title": "Aktuelle Konfiguration:",
        "setup_cfg_provider": "- Anbieter: {}",
        "setup_cfg_key": "- API Key: ✅ gesetzt" if True else "- API Key: ❌ nicht gesetzt",
        "setup_cfg_base_url": "- Base URL: {}",
        "setup_cfg_model": "- Modell: {}",
        "setup_cfg_output_audio": "- Audio-Ausgabe: {}",
        "setup_cfg_output_transcript": "- Transkript-Cache: {}",
        # Tab 1
        "tab1_audio_label": "Podcast-Datei (MP3/WAV)",
        "tab1_theme_label": "Thema für LLM-Suche",
        "tab1_theme_placeholder": "z.B. Buddhismus und Mitgefühl",
        "tab1_provider_label": "API-Anbieter",
        "tab1_model_label": "LLM-Modell",
        "tab1_model_placeholder": "z.B. nvidia/nemotron-3.5-super-vl-120b-a12b",
        "tab1_base_url_label": "Base URL",
        "tab1_base_url_placeholder": "https://integrate.api.nvidia.com/v1",
        "tab1_api_key_label": "API Key",
        "tab1_api_key_placeholder": "nvapi-... oder sk-...",
        "tab1_use_local": "Lokale Unsloth-API verwenden (statt OpenAI Cloud)",
        "tab1_process_btn": "🚀 Transkribieren & Analysieren",
        # Tab 2
        "tab2_transcript_label": "Vollständiges Transkript (mit Zeitcodes)",
        "tab2_passages_label": "Relevante Passagen (markiert)",
        "tab2_confirm_btn": "✅ Schnitt bestätigen",
        "tab2_reject_btn": "❌ Nicht zustimmen / Feedback",
        "tab2_feedback_label": "Feedback (was soll verbessert werden?)",
        "tab2_feedback_placeholder": "z.B. 'Zu wenig Fokus auf Karma' oder 'Passagen 3 & 5 sind zu lang'",
        "tab2_retry_btn": "🔄 Mit Feedback erneut analysieren",
        # Tab 3
        "tab3_audio_label": "Geschnittener Podcast",
        # Callbacks / Messages
        "msg_upload_first": "Bitte laden Sie zuerst eine Audio-Datei hoch.",
        "msg_no_audio_or_passages": "Fehler: Keine Audio-Datei oder keine Passagen vorhanden.",
        "msg_success": "✅ Erfolg! Geschnittener Audio-Track gespeichert unter: {}",
        "msg_cut_error": "Fehler beim Schnitt: {}",
        "msg_http_error": "HTTP-Fehler {} bei der API ({}): {}",
        "msg_connection_error": "Verbindungsabbruch (Versuch {}/{}): {}. Warte 5 Sekunden und versuche erneut...",
        "msg_connection_final": "VERBINDUNGSFEHLER bei der API ({}): {}",
        "msg_api_error": "FEHLER während der API-Anfrage: {}",
        "msg_invalid_json": "FEHLER: LLM gab ungültiges JSON zurück.",
        "msg_raw_response": "Rohantwort: {}",
        "msg_no_relevant_passages": "FEHLER: LLM gab kein erwartetes 'relevant_passages' Array zurück. Antwort: {}",
        # format_passages
        "passage_relevance": "(Relevanz: {}/10)",
        "passage_summary": "→ {}",
        # Errors
        "err_no_passages": "Keine Passagen zum Schneiden vorhanden.",
        # Provider labels
        "provider_nvidia-nim": "NVIDIA NIM",
        "provider_openai": "OpenAI",
        "provider_openrouter": "OpenRouter",
        "provider_unsloth": "Unsloth / vLLM (lokal)",
        "provider_custom": "Eigene / andere API",
    },
    "en": {
        # App
        "app_title": "✂️ PodcastCutter",
        "app_subtitle": "Local audio transcription, LLM-based topic analysis and user-controlled editing. Transcription runs locally via openai-whisper; LLM analysis via a configurable API.",
        "language_label": "Sprache / Language",
        # Tabs
        "tab_setup": "0. 🔧 Setup",
        "tab_setup_done": "0. ✅ Setup",
        "tab_settings": "1. Settings & File",
        "tab_results": "2. Results & Confirmation",
        "tab_cut": "3. Cut & Download",
        # Setup Tab
        "setup_title": "## First-Time Setup",
        "setup_intro": "### Select API provider and enter credentials\n\nChoose the LLM provider you want to use for topic analysis. All OpenAI-compatible APIs are supported.",
        "setup_provider_label": "API Provider",
        "setup_provider_info": "Which LLM API do you want to use?",
        "setup_api_key_label": "API Key",
        "setup_api_key_placeholder": "nvapi-... or sk-...",
        "setup_base_url_label": "Base URL",
        "setup_base_url_placeholder": "https://integrate.api.nvidia.com/v1",
        "setup_model_label": "LLM Model",
        "setup_model_placeholder": "e.g. nvidia/nemotron-3.5-super-vl-120b-a12b",
        "setup_output_audio_label": "Output file for cut MP3",
        "setup_output_audio_placeholder": "/path/to/output.mp3",
        "setup_output_transcript_label": "Output file for transcript cache",
        "setup_output_transcript_placeholder": "/path/to/transcript.json",
        "setup_save_btn": "💾 Save Configuration",
        "setup_save_success": "✅ Configuration saved to .env\nYou can now switch to Tab '1. Settings & File' and use the program.",
        "setup_save_error_missing": "❌ Please fill in all required fields (API Key, Base URL, Model).",
        "setup_already_configured": "Your configuration is already saved (`.env`). You can now use the program.\nTo change settings, edit the `.env` file directly or delete it and restart the program.",
        "setup_show_cfg_btn": "📋 Show Configuration",
        "setup_cfg_title": "Current Configuration:",
        "setup_cfg_provider": "- Provider: {}",
        "setup_cfg_key": "- API Key: ✅ set" if True else "- API Key: ❌ not set",
        "setup_cfg_base_url": "- Base URL: {}",
        "setup_cfg_model": "- Model: {}",
        "setup_cfg_output_audio": "- Audio Output: {}",
        "setup_cfg_output_transcript": "- Transcript Cache: {}",
        # Tab 1
        "tab1_audio_label": "Podcast File (MP3/WAV)",
        "tab1_theme_label": "Topic for LLM Search",
        "tab1_theme_placeholder": "e.g. Buddhism and Compassion",
        "tab1_provider_label": "API Provider",
        "tab1_model_label": "LLM Model",
        "tab1_model_placeholder": "e.g. nvidia/nemotron-3.5-super-vl-120b-a12b",
        "tab1_base_url_label": "Base URL",
        "tab1_base_url_placeholder": "https://integrate.api.nvidia.com/v1",
        "tab1_api_key_label": "API Key",
        "tab1_api_key_placeholder": "nvapi-... or sk-...",
        "tab1_use_local": "Use local Unsloth API (instead of OpenAI Cloud)",
        "tab1_process_btn": "🚀 Transcribe & Analyze",
        # Tab 2
        "tab2_transcript_label": "Full Transcript (with Timecodes)",
        "tab2_passages_label": "Relevant Passages (marked)",
        "tab2_confirm_btn": "✅ Confirm Cut",
        "tab2_reject_btn": "❌ Not accepted / Feedback",
        "tab2_feedback_label": "Feedback (what should be improved?)",
        "tab2_feedback_placeholder": "e.g. 'Too little focus on Karma' or 'Passages 3 & 5 too long'",
        "tab2_retry_btn": "🔄 Analyze again with feedback",
        # Tab 3
        "tab3_audio_label": "Cut Podcast",
        # Callbacks / Messages
        "msg_upload_first": "Please upload an audio file first.",
        "msg_no_audio_or_passages": "Error: No audio file or no passages available.",
        "msg_success": "✅ Success! Cut audio track saved to: {}",
        "msg_cut_error": "Error during cut: {}",
        "msg_http_error": "HTTP Error {} at API ({}): {}",
        "msg_connection_error": "Connection abort (Attempt {}/{}): {}. Waiting 5 seconds and retrying...",
        "msg_connection_final": "CONNECTION ERROR at API ({}): {}",
        "msg_api_error": "ERROR during API request: {}",
        "msg_invalid_json": "ERROR: LLM returned invalid JSON.",
        "msg_raw_response": "Raw response: {}",
        "msg_no_relevant_passages": "ERROR: LLM did not return expected 'relevant_passages' array. Response: {}",
        # format_passages
        "passage_relevance": "(Relevance: {}/10)",
        "passage_summary": "→ {}",
        # Errors
        "err_no_passages": "No passages to cut available.",
        # Provider labels
        "provider_nvidia-nim": "NVIDIA NIM",
        "provider_openai": "OpenAI",
        "provider_openrouter": "OpenRouter",
        "provider_unsloth": "Unsloth / vLLM (local)",
        "provider_custom": "Custom / other API",
    },
}


def t(key: str, lang: str = "de") -> str:
    """Übersetzt einen Schlüssel basierend auf der Sprache."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["de"]).get(key, key)


# ---------------------------------------------------------------------------
# Pfade & Konfigurationsdatei
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(BASE_DIR, ".env")

# Standardwerte (werden aus .env überschrieben, falls vorhanden)
DEFAULT_THEME = "Suchbegriff hier eingeben"
DEFAULT_LLM_MODEL = ""
DEFAULT_BASE_URL = ""
DEFAULT_API_KEY = ""
DEFAULT_API_PROVIDER = "custom"
DEFAULT_OUTPUT_AUDIO = os.path.join(BASE_DIR, "output_trimmed.mp3")
DEFAULT_OUTPUT_TRANSCRIPT = os.path.join(BASE_DIR, "transcript_cache.json")

# Voreingestellte Provider mit deren Standardwerten
PROVIDER_PRESETS = {
    "nvidia-nim": {
        "label_de": "NVIDIA NIM",
        "label_en": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "",
        "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "auth_header": "Bearer",
        "help_de": """### NVIDIA NIM API-Zugang einrichten

1. Gehen Sie auf https://build.nvidia.com/nim und erstellen Sie ein Konto.
2. Suchen Sie nach einem LLM-Modell (z. B. `nemotron-3.5-super-vl-120b-a12b` oder `nemotron-3.5-lightning-30b-a3b`).
3. Erstellen Sie einen API-Key unter "API Keys" in Ihrem NVIDIA NIM Dashboard.
4. Tragen Sie hier Key, Base URL und Modellname ein. Klicken Sie auf "Konfiguration speichern".""",
        "help_en": """### NVIDIA NIM API Setup

1. Go to https://build.nvidia.com/nim and create an account.
2. Search for an LLM model (e.g. `nemotron-3.5-super-vl-120b-a12b` or `nemotron-3.5-lightning-30b-a3b`).
3. Create an API Key under "API Keys" in your NVIDIA NIM Dashboard.
4. Enter Key, Base URL and Model name here. Click "Save Configuration".""",
    },
    "openai": {
        "label_de": "OpenAI",
        "label_en": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini",
        "auth_header": "Bearer",
        "help_de": """### OpenAI API-Zugang einrichten

1. Gehen Sie auf https://platform.openai.com und erstellen Sie ein Konto.
2. Erstellen Sie einen API-Key unter "API Keys" in Ihrem OpenAI Dashboard.
3. Wählen Sie ein Modell (z. B. `gpt-4o-mini`, `gpt-4o`).
4. Tragen Sie hier Key, Base URL und Modellname ein. Klicken Sie auf "Konfiguration speichern".""",
        "help_en": """### OpenAI API Setup

1. Go to https://platform.openai.com and create an account.
2. Create an API Key under "API Keys" in your OpenAI Dashboard.
3. Select a model (e.g. `gpt-4o-mini`, `gpt-4o`).
4. Enter Key, Base URL and Model name here. Click "Save Configuration".""",
    },
    "openrouter": {
        "label_de": "OpenRouter",
        "label_en": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "",
        "model": "openrouter/auto",
        "auth_header": "Bearer",
        "help_de": """### OpenRouter API-Zugang einrichten

1. Gehen Sie auf https://openrouter.ai und erstellen Sie ein Konto.
2. Erstellen Sie einen API-Key unter "Keys" in Ihrem OpenRouter Dashboard.
3. Wählen Sie ein Modell aus dem OpenRouter Model-Katalog.
4. Tragen Sie hier Key, Base URL und Modellname ein. Klicken Sie auf "Konfiguration speichern".""",
        "help_en": """### OpenRouter API Setup

1. Go to https://openrouter.ai and create an account.
2. Create an API Key under "Keys" in your OpenRouter Dashboard.
3. Select a model from the OpenRouter Model Catalog.
4. Enter Key, Base URL and Model name here. Click "Save Configuration".""",
    },
    "unsloth": {
        "label_de": "Unsloth / vLLM (lokal)",
        "label_en": "Unsloth / vLLM (local)",
        "base_url": "http://127.0.0.1:8888/v1",
        "api_key": "not-needed",
        "model": "",
        "auth_header": "Bearer",
        "help_de": """### Lokale Unsloth/vLLM API einrichten

1. Starten Sie einen lokalen Unsloth/vLLM-Server, z. B.:
   `python -m unsloth.vllm.server --model unsloth/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8888`
2. Tragen Sie die Base URL (Standard: `http://127.0.0.1:8888/v1`) ein.
3. Tragen Sie den Modellnamen ein (wie beim Serverstart angegeben).
4. Der API-Key kann für lokale Server meist leer bleiben oder "not-needed" sein.
5. Klicken Sie auf "Konfiguration speichern".""",
        "help_en": """### Local Unsloth/vLLM API Setup

1. Start a local Unsloth/vLLM server, e.g.:
   `python -m unsloth.vllm.server --model unsloth/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8888`
2. Enter the Base URL (default: `http://127.0.0.1:8888/v1`).
3. Enter the model name (as specified at server start).
4. The API key can usually be left empty or set to "not-needed" for local servers.
5. Click "Save Configuration".""",
    },
    "custom": {
        "label_de": "Eigene / andere API",
        "label_en": "Custom / other API",
        "base_url": "",
        "api_key": "",
        "model": "",
        "auth_header": "Bearer",
        "help_de": """### Eigene API einrichten

Tragen Sie Base URL, API-Key und Modellname manuell ein.
Die API muss das OpenAI-kompatible Chat-Completions-Format unterstützen:
`POST {base_url}/chat/completions` mit JSON-Body `{model, messages, temperature}`.""",
        "help_en": """### Custom API Setup

Enter Base URL, API Key and Model name manually.
The API must support the OpenAI-compatible Chat Completions format:
`POST {base_url}/chat/completions` with JSON body `{model, messages, temperature}`.""",
    },
}

# Lade gespeicherte Sprache (Standard: de)
def load_lang() -> str:
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("LANGUAGE="):
                        return line.split("=", 1)[1]
        except Exception:
            pass
    return "de"


def save_lang(lang: str):
    if not lang:
        return
    try:
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    # Entferne alte LANGUAGE-Zeile falls vorhanden
    lines = [l for l in lines if not l.strip().startswith("LANGUAGE=")]
    lines.append(f"LANGUAGE={lang}\n")
    try:
        with open(KEY_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.chmod(KEY_FILE, 0o600)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Konfiguration laden / speichern
# ---------------------------------------------------------------------------
def load_config():
    """Lädt Konfiguration aus .env-Datei. Gibt dict mit allen Keys zurück."""
    config = {
        "API_PROVIDER": DEFAULT_API_PROVIDER,
        "API_KEY": DEFAULT_API_KEY,
        "BASE_URL": DEFAULT_BASE_URL,
        "LLM_MODEL": DEFAULT_LLM_MODEL,
        "OUTPUT_AUDIO": DEFAULT_OUTPUT_AUDIO,
        "OUTPUT_TRANSCRIPT": DEFAULT_OUTPUT_TRANSCRIPT,
    }
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("API_PROVIDER="):
                        config["API_PROVIDER"] = line.split("=", 1)[1]
                    elif line.startswith("API_KEY="):
                        config["API_KEY"] = line.split("=", 1)[1]
                    elif line.startswith("BASE_URL="):
                        config["BASE_URL"] = line.split("=", 1)[1]
                    elif line.startswith("LLM_MODEL="):
                        config["LLM_MODEL"] = line.split("=", 1)[1]
                    elif line.startswith("OUTPUT_AUDIO="):
                        config["OUTPUT_AUDIO"] = line.split("=", 1)[1]
                    elif line.startswith("OUTPUT_TRANSCRIPT="):
                        config["OUTPUT_TRANSCRIPT"] = line.split("=", 1)[1]
        except Exception:
            pass
    return config


def save_config(provider, api_key, base_url, llm_model, output_audio, output_transcript):
    """Speichert Konfiguration in .env-Datei (nur für aktuellen User lesbar)."""
    lines = [
        f"API_PROVIDER={provider}",
        f"API_KEY={api_key}",
        f"BASE_URL={base_url}",
        f"LLM_MODEL={llm_model}",
        f"OUTPUT_AUDIO={output_audio}",
        f"OUTPUT_TRANSCRIPT={output_transcript}",
    ]
    try:
        with open(KEY_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(KEY_FILE, 0o600)
        return True, "Konfiguration gespeichert in .env"
    except Exception as e:
        return False, f"Fehler beim Speichern: {e}"


def config_exists():
    """Prüft, ob .env existiert und einen API-Key enthält."""
    if not os.path.exists(KEY_FILE):
        return False
    cfg = load_config()
    return bool(cfg["API_KEY"].strip())


# ---------------------------------------------------------------------------
# 1. Transkription (lokal, openai-whisper – nutzt ffmpeg, nicht PyAV)
# ---------------------------------------------------------------------------
def transcribe_audio(audio_path: str, model_size: str = "base", language: str = "de"):
    """Transkribiert eine Audio-Datei lokal mit openai-whisper (mit Cache)."""
    cache_path = os.path.splitext(audio_path)[0] + "_transcript_cache.json"

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"Transkript-Cache geladen: {cache_path} ({len(cache['segments'])} Segmente).")
            return cache["segments"], cache.get("language", language)
        except Exception as e:
            print(f"Cache konnte nicht geladen werden, transkribiere neu: {e}")

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language=language, verbose=False)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })

    detected_lang = result.get("language", language)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"segments": segments, "language": detected_lang}, f, ensure_ascii=False, indent=2)
        print(f"Transkript-Cache gespeichert: {cache_path} ({len(segments)} Segmente).")
    except Exception as e:
        print(f"Cache konnte nicht gespeichert werden: {e}")

    return segments, detected_lang


# ---------------------------------------------------------------------------
# 2. LLM-Analyse (beliebige OpenAI-kompatible API)
# ---------------------------------------------------------------------------
def analyze_transcript(segments, theme, llm_model, base_url, api_key, auth_header="Bearer", lang="de"):
    """Sucht mit dem LLM (beliebige OpenAI-kompatible API) nach relevanten Passagen."""
    full_text = "\n".join(
        f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}"
        for seg in segments
    )

    if lang == "en":
        system_prompt = f"""
You are an experienced radio journalist and audio editor.
Your task is to search the following dialog transcript for sections that are thematically related to '{theme}'.

For each relevant section, return a JSON object with the following fields:
1. "segment_index": The index of the segment (0-based) in the original transcript.
2. "start": The start time in seconds (float).
3. "end": The end time in seconds (float).
4. "text": The exact text of the segment.
5. "relevance_score": A number from 1 (low) to 10 (high), how relevant this section is.
6. "summary_german": A very short summary (max 15 words) of the section in German.

The transcript is formatted as follows:
[START_OF_TRANSCRIPT]
[start]s - [end]s: Text content
[start]s - [end]s: Text content
...
[END_OF_TRANSCRIPT]

Answer ONLY with a valid JSON object containing a field "relevant_passages" which is an array of the above objects.
"""
        prompt = f"""Please analyze this transcript and return ONLY a JSON object with relevant sections.

[START_OF_TRANSCRIPT]
{full_text}
[END_OF_TRANSCRIPT]"""
    else:
        system_prompt = f"""
Du bist ein erfahrener Radiojournalist und Audio-Cutter.
Deine Aufgabe ist es, das folgende Dialogtranskript nach Abschnitten zu durchsuchen, die thematisch stark mit '{theme}' in Verbindung stehen.

Für jeden relevanten Abschnitt gibst du ein JSON-Objekt zurück mit folgenden Feldern:
1. "segment_index": Der Index des Segments (0-basiert) im Originaltranskript.
2. "start": Der Startzeitpunkt in Sekunden (float).
3. "end": Der Endzeitpunkt in Sekunden (float).
4. "text": Der exakte Text des Segments.
5. "relevance_score": Eine Zahl von 1 (gering) bis 10 (sehr hoch), wie relevant dieser Abschnitt ist.
6. "summary_german": Eine sehr kurze Zusammenfassung (max 15 Wörter) des Abschnitts auf Deutsch.

Das Transkript ist wie folgt formatiert:
[START_OF_TRANSCRIPT]
[start]s - [end]s: Textinhalt
[start]s - [end]s: Textinhalt
...
[END_OF_TRANSCRIPT]

Antworte NUR mit einem gültigen JSON-Objekt, das ein Feld "relevant_passages" enthält, das ein Array der obigen Objekte ist.
"""
        prompt = f"""Bitte analysiere dieses Transkript und gib NUR ein JSON-Objekt mit relevanten Abschnitten zurück.

[START_OF_TRANSCRIPT]
{full_text}
[END_OF_TRANSCRIPT]"""

    data = json.dumps({
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"{auth_header} {api_key}",
        },
        method="POST",
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = resp.read().decode("utf-8")
            break  # Erfolgreich, Schleife verlassen
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if hasattr(e, 'read') else str(e)
            print(f"HTTP-Fehler {e.code} bei der API ({base_url}): {body[:1000]}")
            return []
        except (urllib.error.URLError, ConnectionResetError, OSError) as e:
            if attempt < max_retries:
                print(f"Verbindungsabbruch (Versuch {attempt}/{max_retries}): {e}. Warte 5 Sekunden und versuche erneut...")
                time.sleep(5)
            else:
                print(f"VERBINDUNGSFEHLER bei der API ({base_url}): {e}")
                return []
        except Exception as e:
            print(f"FEHLER während der API-Anfrage: {e}")
            return []

    try:
        response_wrapper = json.loads(body)
        content = response_wrapper["choices"][0]["message"]["content"]
        analysis_json = json.loads(content)
    except (json.JSONDecodeError, KeyError, IndexError):
        print("FEHLER: LLM gab ungültiges JSON zurück.")
        print(f"Rohantwort: {body[:500]}")
        return []

    if "relevant_passages" in analysis_json and isinstance(analysis_json["relevant_passages"], list):
        normalized = []
        for p in analysis_json["relevant_passages"]:
            normalized.append({
                "segment_index": p.get("segment_index", 0),
                "start": float(p.get("start", 0)),
                "end": float(p.get("end", 0)),
                "text": p.get("text", ""),
                "relevance_score": int(p.get("relevance_score", 0)),
                "summary_german": p.get("summary_german", ""),
            })
        return normalized
    else:
        print(f"FEHLER: LLM gab kein erwartetes 'relevant_passages' Array zurück. Antwort: {body[:200]}")
        return []


# ---------------------------------------------------------------------------
# 3. Audio-Schnitt
# ---------------------------------------------------------------------------
def trim_audio(audio_path: str, relevant_passages: list, output_path: str):
    """Schneidet den Audio-Track auf die markierten Timecodes zu."""
    audio = AudioSegment.from_mp3(audio_path)
    sorted_passages = sorted(relevant_passages, key=lambda x: x["start"])

    if not sorted_passages:
        raise ValueError("Keine Passagen zum Schneiden vorhanden.")

    start_ms = int(sorted_passages[0]["start"] * 1000)
    end_ms = int(sorted_passages[-1]["end"] * 1000)

    trimmed = audio[start_ms:end_ms]
    trimmed.export(output_path, format="mp3")
    return output_path


# ---------------------------------------------------------------------------
# Format-Hilfen für die Anzeige
# ---------------------------------------------------------------------------
def fmt_ts(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    return f"{td.seconds // 60:02d}:{td.seconds % 60:02d}"


def format_transcript(segments):
    lines = []
    for seg in segments:
        lines.append(f"[{fmt_ts(seg['start'])} - {fmt_ts(seg['end'])}] {seg['text']}")
    return "\n".join(lines)


def format_passages(passages, lang="de"):
    lines = []
    for i, rp in enumerate(passages, 1):
        st = fmt_ts(rp.get("start", 0))
        en = fmt_ts(rp.get("end", 0))
        score = rp.get("relevance_score", 0)
        text = rp.get("text", "")
        summary = rp.get("summary_german", "")
        relevance_label = t("passage_relevance", lang).format(score)
        summary_label = t("passage_summary", lang).format(summary)
        lines.append(
            f"{i}. [{st} - {en}] {relevance_label}\n"
            f"{text}\n"
            f"{summary_label}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio-App
# ---------------------------------------------------------------------------
def build_demo():
    cfg = load_config()
    configured = config_exists()
    saved_lang = load_lang()

    provider_choices = list(PROVIDER_PRESETS.keys())

    # Alle UI-Komponenten für Sprachumschaltung speichern
    ui_text_components = []  # (component, attribute, key)

    with gr.Blocks(title="PodcastCutter") as demo:
        # Sprachumschalter oben
        with gr.Row():
            lang_dropdown = gr.Dropdown(
                choices=["de", "en"],
                value=saved_lang,
                label=t("language_label", saved_lang),
            )

        gr.Markdown("")  # Spacer

        # Haupttitel (wird bei Sprachwechsel aktualisiert)
        title_md = gr.Markdown(f"# {t('app_title', saved_lang)}")
        subtitle_md = gr.Markdown(t("app_subtitle", saved_lang))

        # --- Zustand ---------------------------------------------------------
        state_audio_path = gr.State(None)
        state_segments = gr.State(None)
        state_passages = gr.State(None)
        state_theme = gr.State(DEFAULT_THEME)
        state_config = gr.State(cfg)
        state_lang = gr.State(saved_lang)

        def update_language(new_lang):
            save_lang(new_lang)
            updates = []
            for comp, attr, key in ui_text_components:
                new_val = t(key, new_lang)
                if attr == "label":
                    updates.append(gr.update(label=new_val))
                elif attr == "placeholder":
                    updates.append(gr.update(placeholder=new_val))
                elif attr == "value":
                    updates.append(gr.update(value=new_val))
            return updates

        # Hinweis: Sprachumschalter aktualisiert state_lang.
        # Die statischen UI-Texte werden beim Start in der gespeicherten Sprache gerendert.
        # Ein dynamisches Umschalten aller Labels erfordert das Befüllen von ui_text_components,
        # das hier aus Gründen der Einfachheit ausgelassen wird. Die Sprache wird in den
        # Callbacks (on_process, on_retry) korrekt für die Ausgaben verwendet.

        # =====================================================================
        # TAB 0: Einrichtung (Erstlauf)
        # =====================================================================
        tab0_label = t("tab_setup_done", saved_lang) if configured else t("tab_setup", saved_lang)
        with gr.Tab(tab0_label):
            tab0_title_md = gr.Markdown(t("setup_title", saved_lang))
            if not configured:
                tab0_intro_md = gr.Markdown(t("setup_intro", saved_lang))
                setup_provider = gr.Dropdown(
                    choices=provider_choices,
                    value="nvidia-nim",
                    label=t("setup_provider_label", saved_lang),
                    info=t("setup_provider_info", saved_lang),
                )
                setup_help_md = gr.Markdown(PROVIDER_PRESETS["nvidia-nim"]["help_de" if saved_lang == "de" else "help_en"])

                with gr.Row():
                    setup_api_key = gr.Textbox(
                        label=t("setup_api_key_label", saved_lang),
                        type="password",
                        placeholder=t("setup_api_key_placeholder", saved_lang),
                    )
                    setup_base_url = gr.Textbox(
                        label=t("setup_base_url_label", saved_lang),
                        value=PROVIDER_PRESETS["nvidia-nim"]["base_url"],
                        placeholder=t("setup_base_url_placeholder", saved_lang),
                    )
                    setup_llm_model = gr.Textbox(
                        label=t("setup_model_label", saved_lang),
                        value=PROVIDER_PRESETS["nvidia-nim"]["model"],
                        placeholder=t("setup_model_placeholder", saved_lang),
                    )
                with gr.Row():
                    setup_output_audio = gr.Textbox(
                        label=t("setup_output_audio_label", saved_lang),
                        value=DEFAULT_OUTPUT_AUDIO,
                        placeholder=t("setup_output_audio_placeholder", saved_lang),
                    )
                    setup_output_transcript = gr.Textbox(
                        label=t("setup_output_transcript_label", saved_lang),
                        value=DEFAULT_OUTPUT_TRANSCRIPT,
                        placeholder=t("setup_output_transcript_placeholder", saved_lang),
                    )
                setup_save_btn = gr.Button(t("setup_save_btn", saved_lang), variant="primary")
                setup_status = gr.Markdown("")

                def update_help(provider_key, lang):
                    return gr.update(value=PROVIDER_PRESETS[provider_key]["help_de" if lang == "de" else "help_en"])

                setup_provider.change(
                    fn=update_help,
                    inputs=[setup_provider, state_lang],
                    outputs=[setup_help_md],
                )
            else:
                tab0_configured_md = gr.Markdown(t("setup_already_configured", saved_lang))
                with gr.Row():
                    show_cfg_btn = gr.Button(t("setup_show_cfg_btn", saved_lang))
                cfg_display = gr.Markdown("")

        # --- Tab 1: Einstellungen & Verarbeitung ---------------------------
        tab1_label = t("tab_settings", saved_lang)
        with gr.Tab(tab1_label):
            with gr.Row():
                audio_upload = gr.File(label=t("tab1_audio_label", saved_lang), file_types=[".mp3", ".wav"])
                theme_input = gr.Textbox(
                    label=t("tab1_theme_label", saved_lang),
                    value=DEFAULT_THEME,
                    placeholder=t("tab1_theme_placeholder", saved_lang),
                )
            with gr.Row():
                provider_input = gr.Dropdown(
                    choices=provider_choices,
                    value=cfg.get("API_PROVIDER", "custom"),
                    label=t("tab1_provider_label", saved_lang),
                )
                llm_model_input = gr.Textbox(
                    label=t("tab1_model_label", saved_lang),
                    value=cfg.get("LLM_MODEL", ""),
                    placeholder=t("tab1_model_placeholder", saved_lang),
                )
                base_url_input = gr.Textbox(
                    label=t("tab1_base_url_label", saved_lang),
                    value=cfg.get("BASE_URL", ""),
                    placeholder=t("tab1_base_url_placeholder", saved_lang),
                )
                api_key_input = gr.Textbox(
                    label=t("tab1_api_key_label", saved_lang),
                    value=cfg.get("API_KEY", ""),
                    type="password",
                    placeholder=t("tab1_api_key_placeholder", saved_lang),
                )
            process_btn = gr.Button(t("tab1_process_btn", saved_lang), variant="primary")

        # --- Tab 2: Ergebnisse & Bestätigung -------------------------------
        tab2_label = t("tab_results", saved_lang)
        with gr.Tab(tab2_label):
            transcript_output = gr.Textbox(
                label=t("tab2_transcript_label", saved_lang),
                lines=15,
                interactive=False,
            )
            passages_output = gr.Textbox(
                label=t("tab2_passages_label", saved_lang),
                lines=15,
                interactive=False,
            )
            with gr.Row():
                confirm_btn = gr.Button(t("tab2_confirm_btn", saved_lang), variant="primary")
                reject_btn = gr.Button(t("tab2_reject_btn", saved_lang), variant="stop")
            feedback_input = gr.Textbox(
                label=t("tab2_feedback_label", saved_lang),
                placeholder=t("tab2_feedback_placeholder", saved_lang),
            )
            retry_btn = gr.Button(t("tab2_retry_btn", saved_lang))

        # --- Tab 3: Schnitt & Download -------------------------------------
        tab3_label = t("tab_cut", saved_lang)
        with gr.Tab(tab3_label):
            audio_output = gr.Audio(label=t("tab3_audio_label", saved_lang), type="filepath")
            download_info = gr.Markdown("")

        # --- Callbacks -------------------------------------------------------
        def do_setup(provider, api_key, base_url, llm_model, output_audio, output_transcript):
            if not api_key or not base_url or not llm_model:
                return gr.update(value=t("setup_save_error_missing", saved_lang))
            ok, msg = save_config(provider, api_key, base_url, llm_model, output_audio, output_transcript)
            if ok:
                return gr.update(value=f"{t('setup_save_success', saved_lang)}\n")
            else:
                return gr.update(value=f"❌ {msg}")

        def show_configuration():
            c = load_config()
            p = c.get("API_PROVIDER", "custom")
            preset = PROVIDER_PRESETS.get(p, {})
            provider_label = preset.get("label_de" if saved_lang == "de" else "label_en", p)
            key_prefix = "- API Key: ✅ " if c["API_KEY"] else "- API Key: ❌ "
            key_status = "gesetzt" if saved_lang == "de" else "set"
            return gr.update(value=(
                f"{t('setup_cfg_title', saved_lang)}\n"
                f"{t('setup_cfg_provider', saved_lang).format(provider_label)}\n"
                f"{key_prefix}{key_status}\n"
                f"{t('setup_cfg_base_url', saved_lang).format(c['BASE_URL'] or ('nicht gesetzt' if saved_lang == 'de' else 'not set'))}\n"
                f"{t('setup_cfg_model', saved_lang).format(c['LLM_MODEL'] or ('nicht gesetzt' if saved_lang == 'de' else 'not set'))}\n"
                f"{t('setup_cfg_output_audio', saved_lang).format(c['OUTPUT_AUDIO'])}\n"
                f"{t('setup_cfg_output_transcript', saved_lang).format(c['OUTPUT_TRANSCRIPT'])}\n"
            ))

        if not configured:
            setup_save_btn.click(
                fn=do_setup,
                inputs=[setup_provider, setup_api_key, setup_base_url,
                        setup_llm_model, setup_output_audio, setup_output_transcript],
                outputs=[setup_status],
            )
        else:
            show_cfg_btn.click(
                fn=show_configuration,
                outputs=[cfg_display],
            )

        def on_process(audio_file, theme, provider, llm_model, base_url, api_key, current_lang):
            save_config(provider, api_key, base_url, llm_model,
                        cfg["OUTPUT_AUDIO"], cfg["OUTPUT_TRANSCRIPT"])
            if audio_file is None:
                return (
                    t("msg_upload_first", current_lang),
                    "",
                    None,
                    None,
                    None,
                    theme,
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            segments, language = transcribe_audio(audio_file.name)
            transcript_text = format_transcript(segments)

            preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["custom"])
            auth_header = preset.get("auth_header", "Bearer")
            passages = analyze_transcript(
                segments, theme, llm_model, base_url, api_key, auth_header, current_lang
            )
            passages_text = format_passages(passages, current_lang)

            return (
                transcript_text,
                passages_text,
                audio_file.name,
                segments,
                passages,
                theme,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        def on_confirm(audio_path, passages):
            try:
                if audio_path is None or not passages:
                    return (
                        gr.update(visible=False),
                        t("msg_no_audio_or_passages", saved_lang),
                    )
                out_path = cfg["OUTPUT_AUDIO"]
                trim_audio(audio_path, passages, out_path)
                return (
                    gr.update(value=out_path, visible=True),
                    t("msg_success", saved_lang).format(out_path),
                )
            except Exception as e:
                print(f"FEHLER beim Audio-Schnitt: {e}")
                return (
                    gr.update(visible=False),
                    f"{t('msg_cut_error', saved_lang).format(e)}",
                )

        def on_reject(feedback, current_theme):
            new_theme = current_theme
            if feedback.strip():
                new_theme = f"{current_theme} ({feedback.strip()})"
            return (
                "",
                new_theme,
                gr.update(value=new_theme),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        def on_retry(audio_path, theme, provider, llm_model, base_url, api_key, feedback, current_lang):
            if audio_path is None:
                return ("", "", None, None, None, theme)
            segments, _ = transcribe_audio(audio_path)
            effective_theme = theme
            if feedback.strip():
                effective_theme = f"{theme} ({feedback.strip()})"
            preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["custom"])
            auth_header = preset.get("auth_header", "Bearer")
            passages = analyze_transcript(
                segments, effective_theme, llm_model, base_url, api_key, auth_header, current_lang
            )
            return (
                format_transcript(segments),
                format_passages(passages, current_lang),
                audio_path,
                segments,
                passages,
                effective_theme,
            )

        # --- Verkabelung -----------------------------------------------------
        process_btn.click(
            fn=on_process,
            inputs=[audio_upload, theme_input, provider_input,
                    llm_model_input, base_url_input, api_key_input, state_lang],
            outputs=[
                transcript_output, passages_output,
                state_audio_path, state_segments, state_passages, state_theme,
                transcript_output, passages_output,
                confirm_btn, reject_btn, feedback_input, retry_btn,
                audio_output, download_info,
            ],
        )

        confirm_btn.click(
            fn=on_confirm,
            inputs=[state_audio_path, state_passages],
            outputs=[audio_output, download_info],
        )

        reject_btn.click(
            fn=on_reject,
            inputs=[feedback_input, state_theme],
            outputs=[passages_output, state_theme, theme_input, passages_output, retry_btn],
        )

        retry_btn.click(
            fn=on_retry,
            inputs=[state_audio_path, state_theme, provider_input,
                    llm_model_input, base_url_input, api_key_input, feedback_input, state_lang],
            outputs=[
                transcript_output, passages_output,
                state_audio_path, state_segments, state_passages, state_theme,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)