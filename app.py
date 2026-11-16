# app.py
"""
PodcastCutter GUI — Gradio-Interface für lokale Audio-Transkription,
LLM-basierte Themenanalyse über beliebige OpenAI-kompatible APIs und
Benutzer-gesteuerten Audio-Schnitt.

Alle Verarbeitungsschritte laufen lokal (Transkription via openai-whisper).
Die LLM-Analyse erfolgt über eine konfigurierbare API (NVIDIA NIM, OpenAI,
OpenRouter, lokale Unsloth/vLLM oder eigene API).

Mehrsprachig: Zwei vollständige Oberflächen (Deutsch / English) über
gr.TabbedInterface.

Workflow:
  Tab 1: Audio hochladen → Transkribieren (nur Transkription, kein LLM)
         → Automatisch zu Tab 2 wechseln
  Tab 2: Thema eingeben → LLM-Analyse → Passagen anzeigen
         → "Schnitt bestätigen" oder "Erneut suchen"
  Tab 3: Geschnittener Podcast herunterladen

Start:
    python3 app.py
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
# Pfade & Konfigurationsdatei
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(BASE_DIR, ".env")

DEFAULT_THEME_DE = ""
DEFAULT_THEME_EN = ""
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
def trim_audio(audio_path, relevant_passages, output_path):
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
        rel_label = "(Relevanz: {}/10)" if lang == "de" else "(Relevance: {}/10)"
        sum_label = "→ {}" if lang == "de" else "→ {}"
        lines.append(
            f"{i}. [{st} - {en}] {rel_label.format(score)}\n"
            f"{text}\n"
            f"{sum_label.format(summary)}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio-App – deutsche Version
# ---------------------------------------------------------------------------
def build_de():
    cfg = load_config()
    configured = config_exists()

    with gr.Blocks(title="PodcastCutter") as demo:
        gr.Markdown("# ✂️ PodcastCutter")
        gr.Markdown("Lokale Audio-Transkription, LLM-Themenanalyse und benutzergesteuerter Schnitt. "
                    "Transkription läuft lokal via openai-whisper; LLM-Analyse über eine konfigurierbare API.")

        state_audio_path = gr.State(None)
        state_segments = gr.State(None)
        state_passages = gr.State(None)
        state_theme = gr.State(None)

        # Tab 0: Einrichtung
        with gr.Tab("0. 🔧 Einrichtung" if not configured else "0. ✅ Einrichtung"):
            gr.Markdown("## Erste Einrichtung")
            if not configured:
                gr.Markdown("### API-Anbieter auswählen und Zugangsdaten eingeben\n\nWählen Sie den LLM-Anbieter aus, den Sie für die Themenanalyse verwenden möchten. Unterstützt werden alle OpenAI-kompatiblen APIs.")
                setup_provider = gr.Dropdown(
                    choices=list(PROVIDER_PRESETS.keys()),
                    value="nvidia-nim",
                    label="API-Anbieter",
                    info="Welche LLM-API möchten Sie verwenden?",
                )
                setup_help = gr.Markdown(PROVIDER_PRESETS["nvidia-nim"]["help_de"])
                with gr.Row():
                    s_api = gr.Textbox(label="API Key", type="password", placeholder="nvapi-... oder sk-...")
                    s_url = gr.Textbox(label="Base URL", value=PROVIDER_PRESETS["nvidia-nim"]["base_url"], placeholder="https://integrate.api.nvidia.com/v1")
                    s_model = gr.Textbox(label="LLM-Modell", value=PROVIDER_PRESETS["nvidia-nim"]["model"], placeholder="z. B. nvidia/nemotron-3.5-super-vl-120b-a12b")
                with gr.Row():
                    s_out_audio = gr.Textbox(label="Ausgabe-Datei für geschnittenes MP3", value=DEFAULT_OUTPUT_AUDIO, placeholder="/pfad/zum/ausgabe.mp3")
                    s_out_trans = gr.Textbox(label="Ausgabe-Datei für Transkript-Cache", value=DEFAULT_OUTPUT_TRANSCRIPT, placeholder="/pfad/zum/transkript.json")
                s_save = gr.Button("💾 Konfiguration speichern", variant="primary")
                s_status = gr.Markdown("")

                def upd_help(provider):
                    return gr.update(value=PROVIDER_PRESETS[provider]["help_de"])
                setup_provider.change(fn=upd_help, inputs=[setup_provider], outputs=[setup_help])

                def do_setup(prov, key, url, model, out_a, out_t):
                    if not key or not url or not model:
                        return gr.update(value="❌ Bitte füllen Sie alle Pflichtfelder aus (API Key, Base URL, Modell).")
                    ok, msg = save_config(prov, key, url, model, out_a, out_t)
                    if ok:
                        return gr.update(value="✅ Konfiguration gespeichert in .env\nSie können jetzt zu Tab '1. Einstellungen & Datei' wechseln und das Programm nutzen.")
                    return gr.update(value=f"❌ {msg}")

                s_save.click(fn=do_setup,
                             inputs=[setup_provider, s_api, s_url, s_model, s_out_audio, s_out_trans],
                             outputs=[s_status])
            else:
                gr.Markdown("Ihre Konfiguration ist bereits gespeichert (`.env`). Sie können das Programm jetzt nutzen.\nFalls Sie die Einstellungen ändern möchten, bearbeiten Sie die `.env`-Datei direkt oder löschen Sie sie und starten Sie das Programm neu.")
                with gr.Row():
                    show_btn = gr.Button("📋 Konfiguration anzeigen")
                cfg_disp = gr.Markdown("")

                def show_cfg():
                    c = load_config()
                    p = c.get("API_PROVIDER", "custom")
                    pr = PROVIDER_PRESETS.get(p, {}).get("label_de", p)
                    k_ok = "✅ gesetzt" if c["API_KEY"] else "❌ nicht gesetzt"
                    return gr.update(value=(
                        f"Aktuelle Konfiguration:\n"
                        f"- Anbieter: {pr}\n"
                        f"- API Key: {k_ok}\n"
                        f"- Base URL: {c['BASE_URL'] or 'nicht gesetzt'}\n"
                        f"- Modell: {c['LLM_MODEL'] or 'nicht gesetzt'}\n"
                        f"- Audio-Ausgabe: {c['OUTPUT_AUDIO']}\n"
                        f"- Transkript-Cache: {c['OUTPUT_TRANSCRIPT']}\n"
                    ))
                show_btn.click(fn=show_cfg, outputs=[cfg_disp])

        # Tab 1: Audio-Upload & Transkription
        with gr.Tab("1. Einstellungen & Datei"):
            with gr.Row():
                audio_upload = gr.File(label="Podcast-Datei (MP3/WAV)", file_types=[".mp3", ".wav"])
                theme_input = gr.Textbox(label="Thema für LLM-Suche", value=DEFAULT_THEME_DE, placeholder="Hier Thema eingeben")
            with gr.Row():
                p_input = gr.Dropdown(choices=list(PROVIDER_PRESETS.keys()), value=cfg.get("API_PROVIDER", "custom"), label="API-Anbieter")
                m_input = gr.Textbox(label="LLM-Modell", value=cfg.get("LLM_MODEL", ""), placeholder="z. B. nvidia/nemotron-3.5-super-vl-120b-a12b")
                b_input = gr.Textbox(label="Base URL", value=cfg.get("BASE_URL", ""), placeholder="https://integrate.api.nvidia.com/v1")
                k_input = gr.Textbox(label="API Key", value=cfg.get("API_KEY", ""), type="password", placeholder="nvapi-... oder sk-...")
            p_btn = gr.Button("🚀 Transkribieren & Analysieren", variant="primary")

        # Tab 2: Ergebnisse & Bestätigung
        with gr.Tab("2. Ergebnisse & Bestätigung"):
            trans_out = gr.Textbox(label="Vollständiges Transkript (mit Zeitcodes)", lines=15, interactive=False)
            pass_out = gr.Textbox(label="Relevante Passagen (markiert)", lines=15, interactive=False)
            with gr.Row():
                confirm_btn = gr.Button("✅ Schnitt bestätigen", variant="primary")
            f_input = gr.Textbox(label="Feedback (was soll verbessert werden?)", placeholder="z.B. 'Passage X ist zu lang od. Begriff Y kommt nicht vor'")
            r_btn2 = gr.Button("🔄 Mit Feedback erneut analysieren")

        # Tab 3: Schnitt & Download
        with gr.Tab("3. Schnitt & Download"):
            a_out = gr.Audio(label="Geschnittener Podcast", type="filepath")
            d_info = gr.Markdown("")

        # Callbacks
        def on_process(audio_file, theme, prov, model, url, key):
            save_config(prov, key, url, model, cfg["OUTPUT_AUDIO"], cfg["OUTPUT_TRANSCRIPT"])
            if audio_file is None:
                return (
                    "Bitte laden Sie zuerst eine Audio-Datei hoch.",
                    "", None, None, None, theme,
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            segments, language = transcribe_audio(audio_file.name)
            t_text = format_transcript(segments)
            gr.Info("Transkription fertig. Bitte wechseln Sie zu Tab 2 'Ergebnisse & Bestätigung'.")
            preset = PROVIDER_PRESETS.get(prov, PROVIDER_PRESETS["custom"])
            auth = preset.get("auth_header", "Bearer")
            passages = analyze_transcript(segments, theme, model, url, key, auth, "de")
            p_text = format_passages(passages, "de")
            return (
                t_text,
                p_text,
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

        def on_confirm(a_path, passages):
            try:
                if a_path is None or not passages:
                    return gr.update(visible=False), "Fehler: Keine Audio-Datei oder keine Passagen vorhanden."
                out = cfg["OUTPUT_AUDIO"]
                trim_audio(a_path, passages, out)
                return gr.update(value=out, visible=True), f"✅ Erfolg! Geschnittener Audio-Track gespeichert unter: {out}"
            except Exception as e:
                print(f"FEHLER beim Audio-Schnitt: {e}")
                return gr.update(visible=False), f"Fehler beim Schnitt: {e}"

        def on_retry(a_path, theme, prov, model, url, key, feedback):
            if a_path is None:
                return "", "", None, None, None, theme
            segments, _ = transcribe_audio(a_path.name)
            et = theme
            if feedback.strip():
                et = f"{theme} ({feedback.strip()})"
            preset = PROVIDER_PRESETS.get(prov, PROVIDER_PRESETS["custom"])
            auth = preset.get("auth_header", "Bearer")
            passages = analyze_transcript(segments, et, model, url, key, auth, "de")
            return (
                format_transcript(segments),
                format_passages(passages, "de"),
                a_path,
                segments,
                passages,
                et,
            )

        p_btn.click(
            fn=on_process,
            inputs=[audio_upload, theme_input, p_input, m_input, b_input, k_input],
            outputs=[
                trans_out,
                pass_out,
                state_audio_path,
                state_segments,
                state_passages,
                state_theme,
                trans_out,
                pass_out,
                confirm_btn,
                r_btn2,
                a_out,
                d_info,
            ],
        )

        confirm_btn.click(
            fn=on_confirm,
            inputs=[state_audio_path, state_passages],
            outputs=[a_out, d_info],
        )

        r_btn2.click(
            fn=on_retry,
            inputs=[state_audio_path, state_theme, p_input, m_input, b_input, k_input, f_input],
            outputs=[trans_out, pass_out,
                     state_audio_path, state_segments, state_passages, state_theme],
        )

    return demo


# ---------------------------------------------------------------------------
# Gradio-App – englische Version
# ---------------------------------------------------------------------------
def build_en():
    cfg = load_config()
    configured = config_exists()

    with gr.Blocks(title="PodcastCutter") as demo:
        gr.Markdown("# ✂️ PodcastCutter")
        gr.Markdown("Local audio transcription, LLM-based topic analysis and user-controlled editing. "
                    "Transcription runs locally via openai-whisper; LLM analysis via a configurable API.")

        state_audio_path = gr.State(None)
        state_segments = gr.State(None)
        state_passages = gr.State(None)
        state_theme = gr.State(None)

        with gr.Tab("0. 🔧 Setup" if not configured else "0. ✅ Setup"):
            gr.Markdown("## First-Time Setup")
            if not configured:
                gr.Markdown("### Select API provider and enter credentials\n\nChoose the LLM provider you want to use for topic analysis. All OpenAI-compatible APIs are supported.")
                setup_provider = gr.Dropdown(
                    choices=list(PROVIDER_PRESETS.keys()),
                    value="nvidia-nim",
                    label="API Provider",
                    info="Which LLM API do you want to use?",
                )
                setup_help = gr.Markdown(PROVIDER_PRESETS["nvidia-nim"]["help_en"])
                with gr.Row():
                    s_api = gr.Textbox(label="API Key", type="password", placeholder="nvapi-... or sk-...")
                    s_url = gr.Textbox(label="Base URL", value=PROVIDER_PRESETS["nvidia-nim"]["base_url"], placeholder="https://integrate.api.nvidia.com/v1")
                    s_model = gr.Textbox(label="LLM Model", value=PROVIDER_PRESETS["nvidia-nim"]["model"], placeholder="e.g. nvidia/nemotron-3.5-super-vl-120b-a12b")
                with gr.Row():
                    s_out_audio = gr.Textbox(label="Output file for cut MP3", value=DEFAULT_OUTPUT_AUDIO, placeholder="/path/to/output.mp3")
                    s_out_trans = gr.Textbox(label="Output file for transcript cache", value=DEFAULT_OUTPUT_TRANSCRIPT, placeholder="/path/to/transcript.json")
                s_save = gr.Button("💾 Save Configuration", variant="primary")
                s_status = gr.Markdown("")

                def upd_help(provider):
                    return gr.update(value=PROVIDER_PRESETS[provider]["help_en"])
                setup_provider.change(fn=upd_help, inputs=[setup_provider], outputs=[setup_help])

                def do_setup(prov, key, url, model, out_a, out_t):
                    if not key or not url or not model:
                        return gr.update(value="❌ Please fill in all required fields (API Key, Base URL, Model).")
                    ok, msg = save_config(prov, key, url, model, out_a, out_t)
                    if ok:
                        return gr.update(value="✅ Configuration saved to .env\nYou can now switch to Tab '1. Upload audio file' and use the program.")
                    return gr.update(value=f"❌ {msg}")

                s_save.click(fn=do_setup,
                             inputs=[setup_provider, s_api, s_url, s_model, s_out_audio, s_out_trans],
                             outputs=[s_status])
            else:
                gr.Markdown("Your configuration is already saved (`.env`). You can now use the program.\nTo change settings, edit the `.env` file directly or delete it and restart the program.")
                with gr.Row():
                    show_btn = gr.Button("📋 Show Configuration")
                cfg_disp = gr.Markdown("")

                def show_cfg():
                    c = load_config()
                    p = c.get("API_PROVIDER", "custom")
                    pr = PROVIDER_PRESETS.get(p, {}).get("label_en", p)
                    k_ok = "✅ set" if c["API_KEY"] else "❌ not set"
                    return gr.update(value=(
                        f"Current Configuration:\n"
                        f"- Provider: {pr}\n"
                        f"- API Key: {k_ok}\n"
                        f"- Base URL: {c['BASE_URL'] or 'not set'}\n"
                        f"- Model: {c['LLM_MODEL'] or 'not set'}\n"
                        f"- Audio Output: {c['OUTPUT_AUDIO']}\n"
                        f"- Transcript Cache: {c['OUTPUT_TRANSCRIPT']}\n"
                    ))
                show_btn.click(fn=show_cfg, outputs=[cfg_disp])

        with gr.Tab("1. Upload audio file & transcribe"):
            with gr.Row():
                audio_upload = gr.File(label="Podcast File (MP3/WAV)", file_types=[".mp3", ".wav"])
                theme_input = gr.Textbox(label="Topic for LLM Search", value=DEFAULT_THEME_EN, placeholder="Enter topic here")
            with gr.Row():
                p_input = gr.Dropdown(choices=list(PROVIDER_PRESETS.keys()), value=cfg.get("API_PROVIDER", "custom"), label="API Provider")
                m_input = gr.Textbox(label="LLM Model", value=cfg.get("LLM_MODEL", ""), placeholder="e.g. nvidia/nemotron-3.5-super-vl-120b-a12b")
                b_input = gr.Textbox(label="Base URL", value=cfg.get("BASE_URL", ""), placeholder="https://integrate.api.nvidia.com/v1")
                k_input = gr.Textbox(label="API Key", value=cfg.get("API_KEY", ""), type="password", placeholder="nvapi-... or sk-...")
            p_btn = gr.Button("🚀 Transcribe & Analyze", variant="primary")

        with gr.Tab("2. Results & Confirmation"):
            trans_out = gr.Textbox(label="Full Transcript (with Timecodes)", lines=15, interactive=False)
            pass_out = gr.Textbox(label="Relevant Passages (marked)", lines=15, interactive=False)
            with gr.Row():
                c_btn = gr.Button("✅ Confirm Cut", variant="primary")
            f_input = gr.Textbox(label="Feedback (what should be improved?)", placeholder="e.g. 'Passage X is too long or Term Y doesn't come up'")
            r_btn2 = gr.Button("🔄 Analyze again with feedback")

        with gr.Tab("3. Cut & Download"):
            a_out = gr.Audio(label="Cut Podcast", type="filepath")
            d_info = gr.Markdown("")

        def on_process(audio_file, theme, prov, model, url, key):
            save_config(prov, key, url, model, cfg["OUTPUT_AUDIO"], cfg["OUTPUT_TRANSCRIPT"])
            if audio_file is None:
                return (
                    "Please upload an audio file first.",
                    "", None, None, None, theme,
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            segments, language = transcribe_audio(audio_file.name)
            t_text = format_transcript(segments)
            gr.Info("Transcription complete. Please switch to Tab 2 'Results & Confirmation'.")
            preset = PROVIDER_PRESETS.get(prov, PROVIDER_PRESETS["custom"])
            auth = preset.get("auth_header", "Bearer")
            passages = analyze_transcript(segments, theme, model, url, key, auth, "en")
            p_text = format_passages(passages, "en")
            return (
                t_text,
                p_text,
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

        def on_confirm(a_path, passages):
            try:
                if a_path is None or not passages:
                    return gr.update(visible=False), "Error: No audio file or no passages available."
                out = cfg["OUTPUT_AUDIO"]
                trim_audio(a_path, passages, out)
                return gr.update(value=out, visible=True), f"✅ Success! Cut audio track saved to: {out}"
            except Exception as e:
                print(f"FEHLER beim Audio-Schnitt: {e}")
                return gr.update(visible=False), f"Error during cut: {e}"

        def on_retry(a_path, theme, prov, model, url, key, feedback):
            if a_path is None:
                return "", "", None, None, None, theme
            segments, _ = transcribe_audio(a_path.name)
            et = theme
            if feedback.strip():
                et = f"{theme} ({feedback.strip()})"
            preset = PROVIDER_PRESETS.get(prov, PROVIDER_PRESETS["custom"])
            auth = preset.get("auth_header", "Bearer")
            passages = analyze_transcript(segments, et, model, url, key, auth, "en")
            return (
                format_transcript(segments),
                format_passages(passages, "en"),
                a_path,
                segments,
                passages,
                et,
            )

        p_btn.click(
            fn=on_process,
            inputs=[audio_upload, theme_input, p_input, m_input, b_input, k_input],
            outputs=[
                trans_out,
                pass_out,
                state_audio_path,
                state_segments,
                state_passages,
                state_theme,
                trans_out,
                pass_out,
                c_btn,
                r_btn2,
                a_out,
                d_info,
            ],
        )

        c_btn.click(
            fn=on_confirm,
            inputs=[state_audio_path, state_passages],
            outputs=[a_out, d_info],
        )

        r_btn2.click(
            fn=on_retry,
            inputs=[state_audio_path, state_theme, p_input, m_input, b_input, k_input, f_input],
            outputs=[trans_out, pass_out,
                     state_audio_path, state_segments, state_passages, state_theme],
        )

    return demo


# ---------------------------------------------------------------------------
# Haupt-App mit Sprachumschalter (zwei vollständige Oberflächen)
# ---------------------------------------------------------------------------
def build_demo():
    demo_de = build_de()
    demo_en = build_en()

    with gr.Blocks(title="PodcastCutter") as demo:
        gr.Markdown("# ✂️ PodcastCutter")
        gr.Markdown("Wählen Sie Ihre Sprache / Select your language:")

        demo_tabbed = gr.TabbedInterface(
            [demo_de, demo_en],
            ["🇩🇪 Deutsch", "🇬🇧 English"],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)