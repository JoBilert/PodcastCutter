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

TR = {
    "de": {
        "app_title": "✂️ PodcastCutter",
        "app_subtitle": "Lokale Audio-Transkription, LLM-Themenanalyse und benutzergesteuerter Schnitt. Transkription läuft lokal via openai-whisper; LLM-Analyse über eine konfigurierbare API.",
        "lang_label": "Sprache / Language",
        "tab_setup": "0. 🔧 Einrichtung",
        "tab_setup_done": "0. ✅ Einrichtung",
        "tab_settings": "1. Einstellungen & Datei",
        "tab_results": "2. Ergebnisse & Bestätigung",
        "tab_cut": "3. Schnitt & Download",
        "setup_title": "## Erste Einrichtung",
        "setup_intro": "### API-Anbieter auswählen und Zugangsdaten eingeben\n\nWählen Sie den LLM-Anbieter aus, den Sie für die Themenanalyse verwenden möchten. Unterstützt werden alle OpenAI-kompatiblen APIs.",
        "setup_provider": "API-Anbieter",
        "setup_provider_info": "Welche LLM-API möchten Sie verwenden?",
        "setup_api_key": "API Key",
        "setup_api_key_ph": "nvapi-... oder sk-...",
        "setup_base_url": "Base URL",
        "setup_base_url_ph": "https://integrate.api.nvidia.com/v1",
        "setup_model": "LLM-Modell",
        "setup_model_ph": "z. B. nvidia/nemotron-3.5-super-vl-120b-a12b",
        "setup_out_audio": "Ausgabe-Datei für geschnittenes MP3",
        "setup_out_audio_ph": "/pfad/zum/ausgabe.mp3",
        "setup_out_transcript": "Ausgabe-Datei für Transkript-Cache",
        "setup_out_transcript_ph": "/pfad/zum/transkript.json",
        "setup_save": "💾 Konfiguration speichern",
        "setup_save_ok": "✅ Konfiguration gespeichert in .env\nSie können jetzt zu Tab '1. Einstellungen & Datei' wechseln und das Programm nutzen.",
        "setup_save_err": "❌ Bitte füllen Sie alle Pflichtfelder aus (API Key, Base URL, Modell).",
        "setup_configured": "Ihre Konfiguration ist bereits gespeichert (`.env`). Sie können das Programm jetzt nutzen.\nFalls Sie die Einstellungen ändern möchten, bearbeiten Sie die `.env`-Datei direkt oder löschen Sie sie und starten Sie das Programm neu.",
        "setup_show_cfg": "📋 Konfiguration anzeigen",
        "setup_cfg_title": "Aktuelle Konfiguration:",
        "setup_cfg_provider": "- Anbieter: {}",
        "setup_cfg_key_ok": "- API Key: ✅ gesetzt",
        "setup_cfg_key_no": "- API Key: ❌ nicht gesetzt",
        "setup_cfg_base_url": "- Base URL: {}",
        "setup_cfg_model": "- Modell: {}",
        "setup_cfg_out_audio": "- Audio-Ausgabe: {}",
        "setup_cfg_out_transcript": "- Transkript-Cache: {}",
        "setup_help_nim": "### NVIDIA NIM API-Zugang einrichten\n\n1. Gehen Sie auf https://build.nvidia.com/nim und erstellen Sie ein Konto.\n2. Suchen Sie nach einem LLM-Modell (z. B. `nemotron-3.5-super-vl-120b-a12b` oder `nemotron-3.5-lightning-30b-a3b`).\n3. Erstellen Sie einen API-Key unter \"API Keys\" in Ihrem NVIDIA NIM Dashboard.\n4. Tragen Sie hier Key, Base URL und Modellname ein. Klicken Sie auf \"Konfiguration speichern\".",
        "setup_help_openai": "### OpenAI API-Zugang einrichten\n\n1. Gehen Sie auf https://platform.openai.com und erstellen Sie ein Konto.\n2. Erstellen Sie einen API-Key unter \"API Keys\" in Ihrem OpenAI Dashboard.\n3. Wählen Sie ein Modell (z. B. `gpt-4o-mini`, `gpt-4o`).\n4. Tragen Sie hier Key, Base URL und Modellname ein. Klicken Sie auf \"Konfiguration speichern\".",
        "setup_help_openrouter": "### OpenRouter API-Zugang einrichten\n\n1. Gehen Sie auf https://openrouter.ai und erstellen Sie ein Konto.\n2. Erstellen Sie einen API-Key unter \"Keys\" in Ihrem OpenRouter Dashboard.\n3. Wählen Sie ein Modell aus dem OpenRouter Model-Katalog.\n4. Tragen Sie hier Key, Base URL und Modellname ein. Klicken Sie auf \"Konfiguration speichern\".",
        "setup_help_unsloth": "### Lokale Unsloth/vLLM API einrichten\n\n1. Starten Sie einen lokalen Unsloth/vLLM-Server, z. B.:\n   `python -m unsloth.vllm.server --model unsloth/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8888`\n2. Tragen Sie die Base URL (Standard: `http://127.0.0.1:8888/v1`) ein.\n3. Tragen Sie den Modellnamen ein (wie beim Serverstart angegeben).\n4. Der API-Key kann für lokale Server meist leer bleiben oder \"not-needed\" sein.\n5. Klicken Sie auf \"Konfiguration speichern\".",
        "setup_help_custom": "### Eigene API einrichten\n\nTragen Sie Base URL, API-Key und Modellname manuell ein.\nDie API muss das OpenAI-kompatible Chat-Completions-Format unterstützen:\n`POST {base_url}/chat/completions` mit JSON-Body `{model, messages, temperature}`.",
        "t1_audio": "Podcast-Datei (MP3/WAV)",
        "t1_theme": "Thema für LLM-Suche",
        "t1_theme_ph": "z.B. Buddhismus und Mitgefühl",
        "t1_provider": "API-Anbieter",
        "t1_model": "LLM-Modell",
        "t1_model_ph": "z.B. nvidia/nemotron-3.5-super-vl-120b-a12b",
        "t1_base_url": "Base URL",
        "t1_base_url_ph": "https://integrate.api.nvidia.com/v1",
        "t1_api_key": "API Key",
        "t1_api_key_ph": "nvapi-... oder sk-...",
        "t1_use_local": "Lokale Unsloth-API verwenden (statt OpenAI Cloud)",
        "t1_process": "🚀 Transkribieren & Analysieren",
        "t2_transcript": "Vollständiges Transkript (mit Zeitcodes)",
        "t2_passages": "Relevante Passagen (markiert)",
        "t2_confirm": "✅ Schnitt bestätigen",
        "t2_reject": "❌ Nicht zustimmen / Feedback",
        "t2_feedback": "Feedback (was soll verbessert werden?)",
        "t2_feedback_ph": "z.B. 'Zu wenig Fokus auf Karma' oder 'Passagen 3 & 5 sind zu lang'",
        "t2_retry": "🔄 Mit Feedback erneut analysieren",
        "t3_audio": "Geschnittener Podcast",
        "msg_upload": "Bitte laden Sie zuerst eine Audio-Datei hoch.",
        "msg_no_data": "Fehler: Keine Audio-Datei oder keine Passagen vorhanden.",
        "msg_success": "✅ Erfolg! Geschnittener Audio-Track gespeichert unter: {}",
        "msg_cut_error": "Fehler beim Schnitt: {}",
        "msg_http": "HTTP-Fehler {} bei der API ({}): {}",
        "msg_conn": "Verbindungsabbruch (Versuch {}/{}): {}. Warte 5 Sekunden und versuche erneut...",
        "msg_conn_final": "VERBINDUNGSFEHLER bei der API ({}): {}",
        "msg_api_err": "FEHLER während der API-Anfrage: {}",
        "msg_json": "FEHLER: LLM gab ungültiges JSON zurück.",
        "msg_raw": "Rohantwort: {}",
        "msg_no_pass": "FEHLER: LLM gab kein erwartetes 'relevant_passages' Array zurück. Antwort: {}",
        "passage_rel": "(Relevanz: {}/10)",
        "passage_sum": "→ {}",
        "err_no_passages": "Keine Passagen zum Schneiden vorhanden.",
        "prov_nim": "NVIDIA NIM",
        "prov_openai": "OpenAI",
        "prov_openrouter": "OpenRouter",
        "prov_unsloth": "Unsloth / vLLM (lokal)",
        "prov_custom": "Eigene / andere API",
    },
    "en": {
        "app_title": "✂️ PodcastCutter",
        "app_subtitle": "Local audio transcription, LLM-based topic analysis and user-controlled editing. Transcription runs locally via openai-whisper; LLM analysis via a configurable API.",
        "lang_label": "Sprache / Language",
        "tab_setup": "0. 🔧 Setup",
        "tab_setup_done": "0. ✅ Setup",
        "tab_settings": "1. Settings & File",
        "tab_results": "2. Results & Confirmation",
        "tab_cut": "3. Cut & Download",
        "setup_title": "## First-Time Setup",
        "setup_intro": "### Select API provider and enter credentials\n\nChoose the LLM provider you want to use for topic analysis. All OpenAI-compatible APIs are supported.",
        "setup_provider": "API Provider",
        "setup_provider_info": "Which LLM API do you want to use?",
        "setup_api_key": "API Key",
        "setup_api_key_ph": "nvapi-... or sk-...",
        "setup_base_url": "Base URL",
        "setup_base_url_ph": "https://integrate.api.nvidia.com/v1",
        "setup_model": "LLM Model",
        "setup_model_ph": "e.g. nvidia/nemotron-3.5-super-vl-120b-a12b",
        "setup_out_audio": "Output file for cut MP3",
        "setup_out_audio_ph": "/path/to/output.mp3",
        "setup_out_transcript": "Output file for transcript cache",
        "setup_out_transcript_ph": "/path/to/transcript.json",
        "setup_save": "💾 Save Configuration",
        "setup_save_ok": "✅ Configuration saved to .env\nYou can now switch to Tab '1. Settings & File' and use the program.",
        "setup_save_err": "❌ Please fill in all required fields (API Key, Base URL, Model).",
        "setup_configured": "Your configuration is already saved (`.env`). You can now use the program.\nTo change settings, edit the `.env` file directly or delete it and restart the program.",
        "setup_show_cfg": "📋 Show Configuration",
        "setup_cfg_title": "Current Configuration:",
        "setup_cfg_provider": "- Provider: {}",
        "setup_cfg_key_ok": "- API Key: ✅ set",
        "setup_cfg_key_no": "- API Key: ❌ not set",
        "setup_cfg_base_url": "- Base URL: {}",
        "setup_cfg_model": "- Model: {}",
        "setup_cfg_out_audio": "- Audio Output: {}",
        "setup_cfg_out_transcript": "- Transcript Cache: {}",
        "setup_help_nim": "### NVIDIA NIM API Setup\n\n1. Go to https://build.nvidia.com/nim and create an account.\n2. Search for an LLM model (e.g. `nemotron-3.5-super-vl-120b-a12b` or `nemotron-3.5-lightning-30b-a3b`).\n3. Create an API Key under \"API Keys\" in your NVIDIA NIM Dashboard.\n4. Enter Key, Base URL and Model name here. Click \"Save Configuration\".",
        "setup_help_openai": "### OpenAI API Setup\n\n1. Go to https://platform.openai.com and create an account.\n2. Create an API Key under \"API Keys\" in your OpenAI Dashboard.\n3. Select a model (e.g. `gpt-4o-mini`, `gpt-4o`).\n4. Enter Key, Base URL and Model name here. Click \"Save Configuration\".",
        "setup_help_openrouter": "### OpenRouter API Setup\n\n1. Go to https://openrouter.ai and create an account.\n2. Create an API Key under \"Keys\" in your OpenRouter Dashboard.\n3. Select a model from the OpenRouter Model Catalog.\n4. Enter Key, Base URL and Model name here. Click \"Save Configuration\".",
        "setup_help_unsloth": "### Local Unsloth/vLLM API Setup\n\n1. Start a local Unsloth/vLLM server, e.g.:\n   `python -m unsloth.vllm.server --model unsloth/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8888`\n2. Enter the Base URL (default: `http://127.0.0.1:8888/v1`).\n3. Enter the model name (as specified at server start).\n4. The API key can usually be left empty or set to \"not-needed\" for local servers.\n5. Click \"Save Configuration\".",
        "setup_help_custom": "### Custom API Setup\n\nEnter Base URL, API Key and Model name manually.\nThe API must support the OpenAI-compatible Chat Completions format:\n`POST {base_url}/chat/completions` with JSON body `{model, messages, temperature}`.",
        "t1_audio": "Podcast File (MP3/WAV)",
        "t1_theme": "Topic for LLM Search",
        "t1_theme_ph": "e.g. Buddhism and Compassion",
        "t1_provider": "API Provider",
        "t1_model": "LLM Model",
        "t1_model_ph": "e.g. nvidia/nemotron-3.5-super-vl-120b-a12b",
        "t1_base_url": "Base URL",
        "t1_base_url_ph": "https://integrate.api.nvidia.com/v1",
        "t1_api_key": "API Key",
        "t1_api_key_ph": "nvapi-... or sk-...",
        "t1_use_local": "Use local Unsloth API (instead of OpenAI Cloud)",
        "t1_process": "🚀 Transcribe & Analyze",
        "t2_transcript": "Full Transcript (with Timecodes)",
        "t2_passages": "Relevant Passages (marked)",
        "t2_confirm": "✅ Confirm Cut",
        "t2_reject": "❌ Not accepted / Feedback",
        "t2_feedback": "Feedback (what should be improved?)",
        "t2_feedback_ph": "e.g. 'Too little focus on Karma' or 'Passages 3 & 5 too long'",
        "t2_retry": "🔄 Analyze again with feedback",
        "t3_audio": "Cut Podcast",
        "msg_upload": "Please upload an audio file first.",
        "msg_no_data": "Error: No audio file or no passages available.",
        "msg_success": "✅ Success! Cut audio track saved to: {}",
        "msg_cut_error": "Error during cut: {}",
        "msg_http": "HTTP Error {} at API ({}): {}",
        "msg_conn": "Connection abort (Attempt {}/{}): {}. Waiting 5 seconds and retrying...",
        "msg_conn_final": "CONNECTION ERROR at API ({}): {}",
        "msg_api_err": "ERROR during API request: {}",
        "msg_json": "ERROR: LLM returned invalid JSON.",
        "msg_raw": "Raw response: {}",
        "msg_no_pass": "ERROR: LLM did not return expected 'relevant_passages' array. Response: {}",
        "passage_rel": "(Relevance: {}/10)",
        "passage_sum": "→ {}",
        "err_no_passages": "No passages to cut available.",
        "prov_nim": "NVIDIA NIM",
        "prov_openai": "OpenAI",
        "prov_openrouter": "OpenRouter",
        "prov_unsloth": "Unsloth / vLLM (local)",
        "prov_custom": "Custom / other API",
    },
}


def T(key, lang="de"):
    return TR.get(lang, TR["de"]).get(key, key)


def prov_label(key, lang="de"):
    return PROVIDER_PRESETS[key].get(f"label_{lang}", PROVIDER_PRESETS[key]["label_de"])


def prov_help(key, lang="de"):
    return PROVIDER_PRESETS[key].get(f"help_{lang}", PROVIDER_PRESETS[key]["help_de"])


# ---------------------------------------------------------------------------
# Pfade & Konfigurationsdatei
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(BASE_DIR, ".env")

DEFAULT_THEME = "Suchbegriff hier eingeben"
DEFAULT_LLM_MODEL = ""
DEFAULT_BASE_URL = ""
DEFAULT_API_KEY = ""
DEFAULT_API_PROVIDER = "custom"
DEFAULT_OUTPUT_AUDIO = os.path.join(BASE_DIR, "output_trimmed.mp3")
DEFAULT_OUTPUT_TRANSCRIPT = os.path.join(BASE_DIR, "transcript_cache.json")

PROVIDER_PRESETS = {
    "nvidia-nim": {
        "label_de": "NVIDIA NIM",
        "label_en": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "",
        "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "auth_header": "Bearer",
        "help_de": TR["de"]["setup_help_nim"],
        "help_en": TR["en"]["setup_help_nim"],
    },
    "openai": {
        "label_de": "OpenAI",
        "label_en": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini",
        "auth_header": "Bearer",
        "help_de": TR["de"]["setup_help_openai"],
        "help_en": TR["en"]["setup_help_openai"],
    },
    "openrouter": {
        "label_de": "OpenRouter",
        "label_en": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "",
        "model": "openrouter/auto",
        "auth_header": "Bearer",
        "help_de": TR["de"]["setup_help_openrouter"],
        "help_en": TR["en"]["setup_help_openrouter"],
    },
    "unsloth": {
        "label_de": "Unsloth / vLLM (lokal)",
        "label_en": "Unsloth / vLLM (local)",
        "base_url": "http://127.0.0.1:8888/v1",
        "api_key": "not-needed",
        "model": "",
        "auth_header": "Bearer",
        "help_de": TR["de"]["setup_help_unsloth"],
        "help_en": TR["en"]["setup_help_unsloth"],
    },
    "custom": {
        "label_de": "Eigene / andere API",
        "label_en": "Custom / other API",
        "base_url": "",
        "api_key": "",
        "model": "",
        "auth_header": "Bearer",
        "help_de": TR["de"]["setup_help_custom"],
        "help_en": TR["en"]["setup_help_custom"],
    },
}


def load_lang():
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


def save_lang(lang):
    if not lang:
        return
    try:
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    lines = [l for l in lines if not l.strip().startswith("LANGUAGE=")]
    lines.append(f"LANGUAGE={lang}\n")
    try:
        with open(KEY_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.chmod(KEY_FILE, 0o600)
    except Exception:
        pass


def load_config():
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
    if not os.path.exists(KEY_FILE):
        return False
    cfg = load_config()
    return bool(cfg["API_KEY"].strip())


# ---------------------------------------------------------------------------
# 1. Transkription (lokal, openai-whisper)
# ---------------------------------------------------------------------------
def transcribe_audio(audio_path, model_size="base", language="de"):
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
            break
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
def trim_audio(audio_path, relevant_passages, output_path, lang="de"):
    audio = AudioSegment.from_mp3(audio_path)
    sorted_passages = sorted(relevant_passages, key=lambda x: x["start"])
    if not sorted_passages:
        raise ValueError(T("err_no_passages", lang))
    start_ms = int(sorted_passages[0]["start"] * 1000)
    end_ms = int(sorted_passages[-1]["end"] * 1000)
    trimmed = audio[start_ms:end_ms]
    trimmed.export(output_path, format="mp3")
    return output_path


# ---------------------------------------------------------------------------
# Format-Hilfen
# ---------------------------------------------------------------------------
def fmt_ts(seconds):
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
        lines.append(
            f"{i}. [{st} - {en}] {T('passage_rel', lang).format(score)}\n"
            f"{text}\n"
            f"{T('passage_sum', lang).format(summary)}\n"
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

    # Alle UI-Komponenten für Sprachumschaltung: (component, attribute, key_de, key_en, format_fn)
    ui_components = []

    def reg(comp, attr, key_de, key_en=None, fmt=None):
        ui_components.append((comp, attr, key_de, key_en or key_de, fmt))

    with gr.Blocks(title="PodcastCutter") as demo:
        # Sprachumschalter
        with gr.Row():
            lang_dropdown = gr.Dropdown(
                choices=["de", "en"],
                value=saved_lang,
                label=T("lang_label", saved_lang),
            )

        gr.Markdown("")

        title_md = gr.Markdown(f"# {T('app_title', saved_lang)}")
        subtitle_md = gr.Markdown(T("app_subtitle", saved_lang))

        # State
        state_audio_path = gr.State(None)
        state_segments = gr.State(None)
        state_passages = gr.State(None)
        state_theme = gr.State(DEFAULT_THEME)
        state_config = gr.State(cfg)
        state_lang = gr.State(saved_lang)

        def switch_lang(new_lang):
            save_lang(new_lang)
            updates = []
            for comp, attr, key_de, key_en, fmt in ui_components:
                text = T(key_de if new_lang == "de" else key_en, new_lang)
                if fmt:
                    text = text.format(*fmt())
                if attr == "label":
                    updates.append(gr.update(label=text))
                elif attr == "value":
                    updates.append(gr.update(value=text))
                elif attr == "placeholder":
                    updates.append(gr.update(placeholder=text))
                elif attr == "info":
                    updates.append(gr.update(info=text))
            return updates

        # Hinweis: Sprachumschalter aktualisiert state_lang.
        # Die statischen UI-Texte werden beim Start in der gespeicherten Sprache gerendert.
        # Ein dynamisches Umschalten aller Labels erfordert das Befüllen von ui_text_components,
        # das hier aus Gründen der Einfachheit ausgelassen wird. Die Sprache wird in den
        # Callbacks (on_process, on_retry) korrekt für die Ausgaben verwendet.

        # =====================================================================
        # TAB 0: Einrichtung (Erstlauf)
        # =====================================================================
            t0_title = gr.Markdown(T("setup_title", saved_lang))
            reg(t0_title, "value", "setup_title")
            if not configured:
                t0_intro = gr.Markdown(T("setup_intro", saved_lang))
                reg(t0_intro, "value", "setup_intro")
                setup_provider = gr.Dropdown(
                    choices=provider_choices,
                    value="nvidia-nim",
                    label=T("setup_provider", saved_lang),
                    info=T("setup_provider_info", saved_lang),
                )
                reg(setup_provider, "label", "setup_provider")
                reg(setup_provider, "info", "setup_provider_info")
                setup_help = gr.Markdown(prov_help("nvidia-nim", saved_lang))
                reg(setup_help, "value", "setup_help_nim")

                with gr.Row():
                    s_api = gr.Textbox(label=T("setup_api_key", saved_lang), type="password",
                                       placeholder=T("setup_api_key_ph", saved_lang))
                    reg(s_api, "label", "setup_api_key")
                    reg(s_api, "placeholder", "setup_api_key_ph")
                    s_url = gr.Textbox(label=T("setup_base_url", saved_lang),
                                       value=PROVIDER_PRESETS["nvidia-nim"]["base_url"],
                                       placeholder=T("setup_base_url_ph", saved_lang))
                    reg(s_url, "label", "setup_base_url")
                    reg(s_url, "placeholder", "setup_base_url_ph")
                    s_model = gr.Textbox(label=T("setup_model", saved_lang),
                                         value=PROVIDER_PRESETS["nvidia-nim"]["model"],
                                         placeholder=T("setup_model_ph", saved_lang))
                    reg(s_model, "label", "setup_model")
                    reg(s_model, "placeholder", "setup_model_ph")
                with gr.Row():
                    s_out_audio = gr.Textbox(label=T("setup_out_audio", saved_lang),
                                             value=DEFAULT_OUTPUT_AUDIO,
                                             placeholder=T("setup_out_audio_ph", saved_lang))
                    reg(s_out_audio, "label", "setup_out_audio")
                    reg(s_out_audio, "placeholder", "setup_out_audio_ph")
                    s_out_trans = gr.Textbox(label=T("setup_out_transcript", saved_lang),
                                             value=DEFAULT_OUTPUT_TRANSCRIPT,
                                             placeholder=T("setup_out_transcript_ph", saved_lang))
                    reg(s_out_trans, "label", "setup_out_transcript")
                    reg(s_out_trans, "placeholder", "setup_out_transcript_ph")
                s_save = gr.Button(T("setup_save", saved_lang), variant="primary")
                reg(s_save, "label", "setup_save")
                s_status = gr.Markdown("")

                def upd_help(provider, lang):
                    return gr.update(value=prov_help(provider, lang))

                setup_provider.change(fn=upd_help, inputs=[setup_provider, state_lang], outputs=[setup_help])

                def do_setup(prov, key, url, model, out_a, out_t):
                    if not key or not url or not model:
                        return gr.update(value=T("setup_save_err", saved_lang))
                    ok, msg = save_config(prov, key, url, model, out_a, out_t)
                    if ok:
                        return gr.update(value=T("setup_save_ok", saved_lang))
                    return gr.update(value=f"❌ {msg}")

                s_save.click(fn=do_setup,
                             inputs=[setup_provider, s_api, s_url, s_model, s_out_audio, s_out_trans],
                             outputs=[s_status])
            else:
                gr.Markdown(T("setup_configured", saved_lang))
                with gr.Row():
                    show_btn = gr.Button(T("setup_show_cfg", saved_lang))
                    reg(show_btn, "label", "setup_show_cfg")
                cfg_disp = gr.Markdown("")

                def show_cfg():
                    c = load_config()
                    p = c.get("API_PROVIDER", "custom")
                    pr = prov_label(p, saved_lang)
                    k_ok = T("setup_cfg_key_ok", saved_lang) if c["API_KEY"] else T("setup_cfg_key_no", saved_lang)
                    return gr.update(value=(
                        f"{T('setup_cfg_title', saved_lang)}\n"
                        f"{T('setup_cfg_provider', saved_lang).format(pr)}\n"
                        f"{k_ok}\n"
                        f"{T('setup_cfg_base_url', saved_lang).format(c['BASE_URL'] or ('nicht gesetzt' if saved_lang == 'de' else 'not set'))}\n"
                        f"{T('setup_cfg_model', saved_lang).format(c['LLM_MODEL'] or ('nicht gesetzt' if saved_lang == 'de' else 'not set'))}\n"
                        f"{T('setup_cfg_out_audio', saved_lang).format(c['OUTPUT_AUDIO'])}\n"
                        f"{T('setup_cfg_out_transcript', saved_lang).format(c['OUTPUT_TRANSCRIPT'])}\n"
                    ))

                show_btn.click(fn=show_cfg, outputs=[cfg_disp])

        # Tab 1
        with gr.Tab(T("tab_settings", saved_lang)):
            with gr.Row():
                audio_upload = gr.File(label=T("t1_audio", saved_lang), file_types=[".mp3", ".wav"])
                reg(audio_upload, "label", "t1_audio")
                theme_input = gr.Textbox(label=T("t1_theme", saved_lang),
                                         value=DEFAULT_THEME,
                                         placeholder=T("t1_theme_ph", saved_lang))
                reg(theme_input, "label", "t1_theme")
                reg(theme_input, "placeholder", "t1_theme_ph")
            with gr.Row():
                p_input = gr.Dropdown(choices=provider_choices,
                                      value=cfg.get("API_PROVIDER", "custom"),
                                      label=T("t1_provider", saved_lang))
                reg(p_input, "label", "t1_provider")
                m_input = gr.Textbox(label=T("t1_model", saved_lang),
                                     value=cfg.get("LLM_MODEL", ""),
                                     placeholder=T("t1_model_ph", saved_lang))
                reg(m_input, "label", "t1_model")
                reg(m_input, "placeholder", "t1_model_ph")
                b_input = gr.Textbox(label=T("t1_base_url", saved_lang),
                                     value=cfg.get("BASE_URL", ""),
                                     placeholder=T("t1_base_url_ph", saved_lang))
                reg(b_input, "label", "t1_base_url")
                reg(b_input, "placeholder", "t1_base_url_ph")
                k_input = gr.Textbox(label=T("t1_api_key", saved_lang),
                                     value=cfg.get("API_KEY", ""),
                                     type="password",
                                     placeholder=T("t1_api_key_ph", saved_lang))
                reg(k_input, "label", "t1_api_key")
                reg(k_input, "placeholder", "t1_api_key_ph")
            p_btn = gr.Button(T("t1_process", saved_lang), variant="primary")
            reg(p_btn, "label", "t1_process")

        # Tab 2
        with gr.Tab(T("tab_results", saved_lang)):
            trans_out = gr.Textbox(label=T("t2_transcript", saved_lang), lines=15, interactive=False)
            reg(trans_out, "label", "t2_transcript")
            pass_out = gr.Textbox(label=T("t2_passages", saved_lang), lines=15, interactive=False)
            reg(pass_out, "label", "t2_passages")
            with gr.Row():
                c_btn = gr.Button(T("t2_confirm", saved_lang), variant="primary")
                reg(c_btn, "label", "t2_confirm")
                r_btn = gr.Button(T("t2_reject", saved_lang), variant="stop")
                reg(r_btn, "label", "t2_reject")
            f_input = gr.Textbox(label=T("t2_feedback", saved_lang),
                                 placeholder=T("t2_feedback_ph", saved_lang))
            reg(f_input, "label", "t2_feedback")
            reg(f_input, "placeholder", "t2_feedback_ph")
            r_btn2 = gr.Button(T("t2_retry", saved_lang))
            reg(r_btn2, "label", "t2_retry")

        # Tab 3
        with gr.Tab(T("tab_cut", saved_lang)):
            a_out = gr.Audio(label=T("t3_audio", saved_lang), type="filepath")
            reg(a_out, "label", "t3_audio")
            d_info = gr.Markdown("")

        # Callbacks
        def on_process(audio_file, theme, prov, model, url, key, cur_lang):
            save_config(prov, key, url, model, cfg["OUTPUT_AUDIO"], cfg["OUTPUT_TRANSCRIPT"])
            if audio_file is None:
                return (
                    T("msg_upload", cur_lang),
                    "", None, None, None, theme,
                    gr.update(visible=True), gr.update(visible=True),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), gr.update(visible=False),
                )
            segments, _ = transcribe_audio(audio_file.name)
            t_text = format_transcript(segments)
            preset = PROVIDER_PRESETS.get(prov, PROVIDER_PRESETS["custom"])
            auth = preset.get("auth_header", "Bearer")
            passages = analyze_transcript(segments, theme, model, url, key, auth, cur_lang)
            p_text = format_passages(passages, cur_lang)
            return (
                t_text, p_text, audio_file.name, segments, passages, theme,
                gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=False),
            )

        def on_confirm(a_path, passages, cur_lang):
            try:
                if a_path is None or not passages:
                    return gr.update(visible=False), T("msg_no_data", cur_lang)
                out = cfg["OUTPUT_AUDIO"]
                trim_audio(a_path, passages, out, cur_lang)
                return gr.update(value=out, visible=True), T("msg_success", cur_lang).format(out)
            except Exception as e:
                print(f"FEHLER beim Audio-Schnitt: {e}")
                return gr.update(visible=False), f"{T('msg_cut_error', cur_lang).format(e)}"

        def on_reject(feedback, cur_theme):
            nt = cur_theme
            if feedback.strip():
                nt = f"{cur_theme} ({feedback.strip()})"
            return "", nt, gr.update(value=nt), gr.update(visible=True), gr.update(visible=False)

        def on_retry(a_path, theme, prov, model, url, key, feedback, cur_lang):
            if a_path is None:
                return "", "", None, None, None, theme
            segments, _ = transcribe_audio(a_path.name)
            et = theme
            if feedback.strip():
                et = f"{theme} ({feedback.strip()})"
            preset = PROVIDER_PRESETS.get(prov, PROVIDER_PRESETS["custom"])
            auth = preset.get("auth_header", "Bearer")
            passages = analyze_transcript(segments, et, model, url, key, auth, cur_lang)
            return (
                format_transcript(segments),
                format_passages(passages, cur_lang),
                a_path, segments, passages, et,
            )

        p_btn.click(fn=on_process,
                    inputs=[audio_upload, theme_input, p_input, m_input, b_input, k_input, state_lang],
                    outputs=[trans_out, pass_out,
                             state_audio_path, state_segments, state_passages, state_theme,
                             trans_out, pass_out,
                             c_btn, r_btn, f_input, r_btn2,
                             a_out, d_info])

        c_btn.click(fn=on_confirm,
                    inputs=[state_audio_path, state_passages, state_lang],
                    outputs=[a_out, d_info])

        r_btn.click(fn=on_reject,
                    inputs=[f_input, state_theme],
                    outputs=[pass_out, state_theme, theme_input, pass_out, r_btn2])

        r_btn2.click(fn=on_retry,
                     inputs=[state_audio_path, state_theme, p_input, m_input, b_input, k_input, f_input, state_lang],
                     outputs=[trans_out, pass_out,
                              state_audio_path, state_segments, state_passages, state_theme])

        # Sprachumschalter an alle registrierten UI-Komponenten binden (nach deren Erstellung)
        lang_dropdown.change(
            fn=switch_lang,
            inputs=[lang_dropdown],
            outputs=[comp for comp, _, _, _, _ in ui_components],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)