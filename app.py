# app.py
"""
PodcastCutter GUI — Gradio-Interface für lokale Audio-Transkription,
LLM-basierte Themenanalyse (Unsloth/vLLM) und Benutzer-gesteuerten Audio-Schnitt.

Alle Verarbeitungsschritte laufen lokal. Es werden keine Online-Dienste benötigt.

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
# Standard-Konfiguration (kann über die GUI überschrieben werden)
# ---------------------------------------------------------------------------
DEFAULT_THEME = "Suchbegriff hier eingeben"
DEFAULT_LLM_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
USE_LOCAL_UNSLOTH = False
OUTPUT_TRIMMED = "output_trimmed.mp3"

KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_key():
    """Lädt den gespeicherten API-Key aus der .env-Datei (falls vorhanden)."""
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "r") as f:
                for line in f:
                    if line.startswith("NIM_API_KEY="):
                        return line.strip().split("=", 1)[1]
        except Exception:
            pass
    return ""


def save_key(key):
    """Speichert den API-Key in der .env-Datei (nur für den aktuellen User lesbar)."""
    if not key:
        return
    try:
        with open(KEY_FILE, "w") as f:
            f.write(f"NIM_API_KEY={key}\n")
        os.chmod(KEY_FILE, 0o600)
    except Exception:
        pass


DEFAULT_API_KEY = load_key()


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
# 2. LLM-Analyse (lokal über Unsloth/vLLM API)
# ---------------------------------------------------------------------------
def analyze_transcript(segments, theme, llm_model, base_url, api_key, use_local):
    """Sucht mit dem LLM (lokal über Unsloth) nach thematisch relevanten Passagen."""
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
            "Authorization": f"Bearer {api_key}",
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
            print(f"HTTP-Fehler {e.code} bei der NVIDIA NIM API ({base_url}): {body[:1000]}")
            return []
        except (urllib.error.URLError, ConnectionResetError, OSError) as e:
            if attempt < max_retries:
                print(f"Verbindungsabbruch (Versuch {attempt}/{max_retries}): {e}. Warte 5 Sekunden und versuche erneut...")
                time.sleep(5)
            else:
                print(f"VERBINDUNGSFEHLER bei der NVIDIA NIM API ({base_url}): {e}")
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
def trim_audio(audio_path: str, relevant_passages: list, output_path: str = OUTPUT_TRIMMED):
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
    with gr.Blocks(title="PodcastCutter") as demo:
        gr.Markdown("# ✂️ PodcastCutter")
        gr.Markdown("Lokale Audio-Transkription, LLM-Themenanalyse und benutzergesteuerter Schnitt. "
                    "Alles läuft offline auf deinem Rechner.")

        # --- Zuständige State-Variablen ------------------------------------
        state_audio_path = gr.State(None)
        state_segments = gr.State(None)
        state_passages = gr.State(None)
        state_theme = gr.State(DEFAULT_THEME)

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
                llm_model_input = gr.Textbox(
                    label="LLM-Modell",
                    value=DEFAULT_LLM_MODEL,
                )
                base_url_input = gr.Textbox(
                    label="Unsloth/vLLM Base URL",
                    value=DEFAULT_BASE_URL,
                )
                api_key_input = gr.Textbox(
                    label="NVIDIA NIM API Key",
                    value=DEFAULT_API_KEY,
                    type="password",
                    placeholder="nvapi-...",
                )
            use_local_checkbox = gr.Checkbox(
                label="Lokale Unsloth-API verwenden (statt OpenAI Cloud)",
                value=False,
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

        # --- Callbacks ------------------------------------------------------
        def on_process(audio_file, theme, llm_model, base_url, api_key, use_local):
            save_key(api_key)
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
                    gr.update(visible=False),
                )

            # Transkription (lokal)
            segments, language = transcribe_audio(audio_file.name)
            transcript_text = format_transcript(segments)

            # LLM-Analyse (lokal über Unsloth)
            passages = analyze_transcript(
                segments, theme, llm_model, base_url, api_key, use_local
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
                out_path = trim_audio(audio_path, passages)
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

        def on_retry(audio_path, theme, llm_model, base_url, api_key, use_local, feedback):
            if audio_path is None:
                return ("", "", None, None, None, theme)
            segments, _ = transcribe_audio(audio_path)
            effective_theme = theme
            if feedback.strip():
                effective_theme = f"{theme} ({feedback.strip()})"
            passages = analyze_transcript(
                segments, effective_theme, llm_model, base_url, api_key, use_local
            )
            return (
                format_transcript(segments),
                format_passages(passages),
                audio_path,
                segments,
                passages,
                effective_theme,
            )

        # --- Verkabelung ----------------------------------------------------
        process_btn.click(
            fn=on_process,
            inputs=[audio_upload, theme_input, llm_model_input,
                    base_url_input, api_key_input, use_local_checkbox],
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
            inputs=[state_audio_path, state_theme, llm_model_input,
                    base_url_input, api_key_input, use_local_checkbox, feedback_input],
            outputs=[
                transcript_output, passages_output,
                state_audio_path, state_segments, state_passages, state_theme,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)