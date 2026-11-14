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
        "label": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "",
        "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "auth_header": "Bearer",
        "help": """
### NVIDIA NIM API-Zugang einrichten

1. Gehen Sie auf https://build.nvidia.com/nim und erstellen Sie ein Konto.
2. Suchen Sie nach einem LLM-Modell (z. B. `nemotron-3.5-super-vl-120b-a12b` oder `nemotron-3.5-lightning-30b-a3b`).
3. Erstellen Sie einen API-Key unter "API Keys" in Ihrem NVIDIA NIM Dashboard.
4. Tragen Sie hier Key, Base URL und Modellname ein. Klicken Sie auf "Konfiguration speichern".
""",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini",
        "auth_header": "Bearer",
        "help": """
### OpenAI API-Zugang einrichten

1. Gehen Sie auf https://platform.openai.com und erstellen Sie ein Konto.
2. Erstellen Sie einen API-Key unter "API Keys" in Ihrem OpenAI Dashboard.
3. Wählen Sie ein Modell (z. B. `gpt-4o-mini`, `gpt-4o`).
4. Tragen Sie hier Key, Base URL und Modellname ein. Klicken Sie auf "Konfiguration speichern".
""",
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "",
        "model": "openrouter/auto",
        "auth_header": "Bearer",
        "help": """
### OpenRouter API-Zugang einrichten

1. Gehen Sie auf https://openrouter.ai und erstellen Sie ein Konto.
2. Erstellen Sie einen API-Key unter "Keys" in Ihrem OpenRouter Dashboard.
3. Wählen Sie ein Modell aus dem OpenRouter Model-Katalog.
4. Tragen Sie hier Key, Base URL und Modellname ein. Klicken Sie auf "Konfiguration speichern".
""",
    },
    "unsloth": {
        "label": "Unsloth / vLLM (lokal)",
        "base_url": "http://127.0.0.1:8888/v1",
        "api_key": "not-needed",
        "model": "",
        "auth_header": "Bearer",
        "help": """
### Lokale Unsloth/vLLM API einrichten

1. Starten Sie einen lokalen Unsloth/vLLM-Server, z. B.:
   `python -m unsloth.vllm.server --model unsloth/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8888`
2. Tragen Sie die Base URL (Standard: `http://127.0.0.1:8888/v1`) ein.
3. Tragen Sie den Modellnamen ein (wie beim Serverstart angegeben).
4. Der API-Key kann für lokale Server meist leer bleiben oder "not-needed" sein.
5. Klicken Sie auf "Konfiguration speichern".
""",
    },
    "custom": {
        "label": "Eigene / andere API",
        "base_url": "",
        "api_key": "",
        "model": "",
        "auth_header": "Bearer",
        "help": """
### Eigene API einrichten

Tragen Sie Base URL, API-Key und Modellname manuell ein.
Die API muss das OpenAI-kompatible Chat-Completions-Format unterstützen:
`POST {base_url}/chat/completions` mit JSON-Body `{model, messages, temperature}`.
""",
    },
}


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
def analyze_transcript(segments, theme, llm_model, base_url, api_key, auth_header="Bearer"):
    """Sucht mit dem LLM (beliebige OpenAI-kompatible API) nach relevanten Passagen."""
    full_text = "\n".join(
        f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}"
        for seg in segments
    )

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
        # Normalize passages: ensure all expected keys exist with defaults
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


def format_passages(passages):
    lines = []
    for i, rp in enumerate(passages, 1):
        st = fmt_ts(rp.get("start", 0))
        en = fmt_ts(rp.get("end", 0))
        score = rp.get("relevance_score", 0)
        text = rp.get("text", "")
        summary = rp.get("summary_german", "")
        lines.append(
            f"{i}. [{st} - {en}] (Relevanz: {score}/10)\n"
            f"{text}\n"
            f"→ {summary}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio-App
# ---------------------------------------------------------------------------
def build_demo():
    cfg = load_config()
    configured = config_exists()

    provider_choices = list(PROVIDER_PRESETS.keys())
    provider_labels = [PROVIDER_PRESETS[p]["label"] for p in provider_choices]

    with gr.Blocks(title="PodcastCutter") as demo:
        gr.Markdown("# ✂️ PodcastCutter")
        gr.Markdown("Lokale Audio-Transkription, LLM-Themenanalyse und benutzergesteuerter Schnitt. "
                    "Transkription läuft lokal via openai-whisper; LLM-Analyse über eine konfigurierbare API.")

        # --- Zustand ---------------------------------------------------------
        state_audio_path = gr.State(None)
        state_segments = gr.State(None)
        state_passages = gr.State(None)
        state_theme = gr.State(DEFAULT_THEME)
        state_config = gr.State(cfg)

        # =====================================================================
        # TAB 0: Einrichtung (Erstlauf)
        # =====================================================================
        with gr.Tab("0. 🔧 Einrichtung" if not configured else "0. ✅ Einrichtung"):
            gr.Markdown("## Erste Einrichtung")
            if not configured:
                gr.Markdown("""
### API-Anbieter auswählen und Zugangsdaten eingeben

Wählen Sie den LLM-Anbieter aus, den Sie für die Themenanalyse verwenden möchten.
Unterstützt werden alle OpenAI-kompatiblen APIs.
""")
                setup_provider = gr.Dropdown(
                    choices=provider_choices,
                    value="nvidia-nim",
                    label="API-Anbieter",
                    info="Welche LLM-API möchten Sie verwenden?",
                )
                setup_help_md = gr.Markdown(PROVIDER_PRESETS["nvidia-nim"]["help"])

                with gr.Row():
                    setup_api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="nvapi-... oder sk-...",
                    )
                    setup_base_url = gr.Textbox(
                        label="Base URL",
                        value=PROVIDER_PRESETS["nvidia-nim"]["base_url"],
                        placeholder="https://integrate.api.nvidia.com/v1",
                    )
                    setup_llm_model = gr.Textbox(
                        label="LLM-Modell",
                        value=PROVIDER_PRESETS["nvidia-nim"]["model"],
                        placeholder="z. B. nvidia/nemotron-3.5-super-vl-120b-a12b",
                    )
                with gr.Row():
                    setup_output_audio = gr.Textbox(
                        label="Ausgabe-Datei für geschnittenes MP3",
                        value=DEFAULT_OUTPUT_AUDIO,
                        placeholder="/pfad/zum/ausgabe.mp3",
                    )
                    setup_output_transcript = gr.Textbox(
                        label="Ausgabe-Datei für Transkript-Cache",
                        value=DEFAULT_OUTPUT_TRANSCRIPT,
                        placeholder="/pfad/zum/transkript.json",
                    )
                setup_save_btn = gr.Button("💾 Konfiguration speichern", variant="primary")
                setup_status = gr.Markdown("")

                def update_help(provider_key):
                    return gr.update(value=PROVIDER_PRESETS[provider_key]["help"])

                setup_provider.change(
                    fn=update_help,
                    inputs=[setup_provider],
                    outputs=[setup_help_md],
                )
            else:
                gr.Markdown("""
Ihre Konfiguration ist bereits gespeichert (`.env`). Sie können das Programm jetzt nutzen.
Falls Sie die Einstellungen ändern möchten, bearbeiten Sie die `.env`-Datei direkt oder löschen Sie sie und starten Sie das Programm neu.
""")
                with gr.Row():
                    show_cfg_btn = gr.Button("📋 Konfiguration anzeigen")
                cfg_display = gr.Markdown("")

        # --- Tab 1: Einstellungen & Verarbeitung ---------------------------
        with gr.Tab("1. Einstellungen & Datei"):
            with gr.Row():
                audio_upload = gr.File(label="Podcast-Datei (MP3/WAV)", file_types=[".mp3", ".wav"])
                theme_input = gr.Textbox(
                    label="Thema für LLM-Suche",
                    value=DEFAULT_THEME,
                    placeholder="z.B. Buddhismus und Mitgefühl",
                )
            with gr.Row():
                provider_input = gr.Dropdown(
                    choices=provider_choices,
                    value=cfg.get("API_PROVIDER", "custom"),
                    label="API-Anbieter",
                )
                llm_model_input = gr.Textbox(
                    label="LLM-Modell",
                    value=cfg.get("LLM_MODEL", ""),
                    placeholder="z.B. nvidia/nemotron-3.5-super-vl-120b-a12b",
                )
                base_url_input = gr.Textbox(
                    label="Base URL",
                    value=cfg.get("BASE_URL", ""),
                    placeholder="https://integrate.api.nvidia.com/v1",
                )
                api_key_input = gr.Textbox(
                    label="API Key",
                    value=cfg.get("API_KEY", ""),
                    type="password",
                    placeholder="nvapi-... oder sk-...",
                )
            process_btn = gr.Button("🚀 Transkribieren & Analysieren", variant="primary")

        # --- Tab 2: Ergebnisse & Bestätigung -------------------------------
        with gr.Tab("2. Ergebnisse & Bestätigung"):
            transcript_output = gr.Textbox(
                label="Vollständiges Transkript (mit Zeitcodes)",
                lines=15,
                interactive=False,
            )
            passages_output = gr.Textbox(
                label="Relevante Passagen (markiert)",
                lines=15,
                interactive=False,
            )
            with gr.Row():
                confirm_btn = gr.Button("✅ Schnitt bestätigen", variant="primary")
                reject_btn = gr.Button("❌ Nicht zustimmen / Feedback", variant="stop")
            feedback_input = gr.Textbox(
                label="Feedback (was soll verbessert werden?)",
                placeholder="z.B. 'Zu wenig Fokus auf Karma' oder 'Passagen 3 & 5 sind zu lang'",
            )
            retry_btn = gr.Button("🔄 Mit Feedback erneut analysieren")

        # --- Tab 3: Schnitt & Download -------------------------------------
        with gr.Tab("3. Schnitt & Download"):
            audio_output = gr.Audio(label="Geschnittener Podcast", type="filepath")
            download_info = gr.Markdown("")

        # --- Callbacks -------------------------------------------------------
        def do_setup(provider, api_key, base_url, llm_model, output_audio, output_transcript):
            if not api_key or not base_url or not llm_model:
                return gr.update(value="❌ Bitte füllen Sie alle Pflichtfelder aus (API Key, Base URL, Modell).")
            ok, msg = save_config(provider, api_key, base_url, llm_model, output_audio, output_transcript)
            if ok:
                return gr.update(value=f"✅ {msg}\nSie können jetzt zu Tab '1. Einstellungen & Datei' wechseln und das Programm nutzen.")
            else:
                return gr.update(value=f"❌ {msg}")

        def show_configuration():
            c = load_config()
            p = c.get("API_PROVIDER", "custom")
            return gr.update(value=f"""Aktuelle Konfiguration:
- Anbieter: {PROVIDER_PRESETS.get(p, {}).get('label', p)}
- API Key: {'✅ gesetzt' if c['API_KEY'] else '❌ nicht gesetzt'}
- Base URL: {c['BASE_URL'] or 'nicht gesetzt'}
- Modell: {c['LLM_MODEL'] or 'nicht gesetzt'}
- Audio-Ausgabe: {c['OUTPUT_AUDIO']}
- Transkript-Cache: {c['OUTPUT_TRANSCRIPT']}
""")

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

        def on_process(audio_file, theme, provider, llm_model, base_url, api_key):
            save_config(provider, api_key, base_url, llm_model,
                        cfg["OUTPUT_AUDIO"], cfg["OUTPUT_TRANSCRIPT"])
            if audio_file is None:
                return (
                    "Bitte laden Sie zuerst eine Audio-Datei hoch.",
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

            # Transkription (lokal)
            segments, language = transcribe_audio(audio_file.name)
            transcript_text = format_transcript(segments)

            # LLM-Analyse (gewählte API)
            preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["custom"])
            auth_header = preset.get("auth_header", "Bearer")
            passages = analyze_transcript(
                segments, theme, llm_model, base_url, api_key, auth_header
            )
            passages_text = format_passages(passages)

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
                        "Fehler: Keine Audio-Datei oder keine Passagen vorhanden.",
                    )
                out_path = cfg["OUTPUT_AUDIO"]
                trim_audio(audio_path, passages, out_path)
                return (
                    gr.update(value=out_path, visible=True),
                    f"✅ Erfolg! Geschnittener Audio-Track gespeichert unter: {out_path}",
                )
            except Exception as e:
                print(f"FEHLER beim Audio-Schnitt: {e}")
                return (
                    gr.update(visible=False),
                    f"Fehler beim Schnitt: {e}",
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

        def on_retry(audio_path, theme, provider, llm_model, base_url, api_key, feedback):
            if audio_path is None:
                return ("", "", None, None, None, theme)
            segments, _ = transcribe_audio(audio_path)
            effective_theme = theme
            if feedback.strip():
                effective_theme = f"{theme} ({feedback.strip()})"
            preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["custom"])
            auth_header = preset.get("auth_header", "Bearer")
            passages = analyze_transcript(
                segments, effective_theme, llm_model, base_url, api_key, auth_header
            )
            return (
                format_transcript(segments),
                format_passages(passages),
                audio_path,
                segments,
                passages,
                effective_theme,
            )

        # --- Verkabelung -----------------------------------------------------
        process_btn.click(
            fn=on_process,
            inputs=[audio_upload, theme_input, provider_input,
                    llm_model_input, base_url_input, api_key_input],
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
                    llm_model_input, base_url_input, api_key_input, feedback_input],
            outputs=[
                transcript_output, passages_output,
                state_audio_path, state_segments, state_passages, state_theme,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)