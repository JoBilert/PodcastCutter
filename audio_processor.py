# audio_processor.py

import os
from pydub import AudioSegment
import openai
from datetime import timedelta
import json

# ==============================================================================
# KONFIGURATION
# ==============================================================================

# Audio-Pfade
AUDIO_INPUT_PATH = "input_dialog.mp3"
AUDIO_OUTPUT_TRIMMED_PATH = "output_trimmed.mp3"

# Thema für die LLM-Suche
THEME_TO_SEARCH = "Buddhismus und Mitgefühl"

# LLM-Modellname (wird an die Unsloth/vLLM API übergeben)
LLM_MODEL = "unsloth/Qwen2.5-7B-Instruct"  # Beispiel; anpassen an dein geladenes Modell

# Unsloth API Endpunkt (lokal laufender vLLM/Unsloth Server)
USE_LOCAL_UNSLOTH = True          # True = lokale Unsloth-API, False = OpenAI Cloud
UNSLOTH_BASE_URL = "http://localhost:8000/v1"  # Standard-Port von Unsloth/vLLM
UNSLOTH_API_KEY = "dummy"         # Bei lokaler API meist egal, aber das SDK verlangt einen String

# OpenAI API Key (nur benötigt wenn USE_LOCAL_UNSLOTH = False)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================

def get_llm_client():
    """Gibt einen OpenAI-kompatiblen Client zurück, der auf die lokale Unsloth-API
    oder auf die OpenAI-Cloud zeigt, je nach USE_LOCAL_UNSLOTH."""
    if USE_LOCAL_UNSLOTH:
        return openai.OpenAI(
            base_url=UNSLOTH_BASE_URL,
            api_key=UNSLOTH_API_KEY
        )
    else:
        return openai.OpenAI(
            api_key=OPENAI_API_KEY
        )


def load_audio(path: str) -> AudioSegment:
    """Lädt die Audio-Datei."""
    try:
        audio = AudioSegment.from_mp3(path)
        print(f"Audio erfolgreich geladen: {path}. Dauer: {len(audio) / 1000.0:.2f} Sekunden.")
        return audio
    except FileNotFoundError:
        print(f"FEHLER: Audio-Datei nicht gefunden unter {path}")
        return None
    except Exception as e:
        print(f"FEHLER beim Laden der Audio-Datei: {e}")
        return None


# ==============================================================================
# 1. TRANSKRIPTION (lokal mit faster-whisper)
# ==============================================================================

def transcribe_audio(audio: AudioSegment) -> list[dict]:
    """Transkribiert Audio lokal mit faster-whisper und liefert Zeitstempel."""
    print("--- Starte lokale Transkription (faster-whisper) ---")

    # Temporäre WAV-Datei für faster-whisper
    temp_wav_path = "temp_transcribe.wav"
    audio.export(temp_wav_path, format="wav")

    try:
        from faster_whisper import WhisperModel

        # Modell laden (CPU oder CUDA)
        # device="cpu" oder device="cuda", compute_type="int8" für CPU, "float16" für GPU
        model = WhisperModel("base", device="cpu", compute_type="int8")

        segments, info = model.transcribe(temp_wav_path, beam_size=5, language="de")

        result = []
        for seg in segments:
            result.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })

        print(f"Transkription erfolgreich: {len(result)} Segmente erkannt (Sprache: {info.language}).")
        return result

    except ImportError:
        print("FEHLER: 'faster-whisper' ist nicht installiert. Bitte 'pip install faster-whisper' ausführen.")
        return []
    except Exception as e:
        print(f"FEHLER während der lokalen Transkription: {e}")
        return []
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


# ==============================================================================
# 2. LLM-ANALYSE (über Unsloth API oder OpenAI)
# ==============================================================================

def analyze_transcript(segments: list[dict], theme: str) -> list[dict]:
    """Sucht mit dem LLM (lokal über Unsloth oder OpenAI) nach relevanten Passagen."""
    print(f"\n--- Starte LLM-Analyse für Thema: '{theme}' ---")

    full_text = "\n".join([
        f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}"
        for seg in segments
    ])

    system_prompt = f"""
Du bist ein erfahrener Religionslehrer für die 10. Klasse (Bayern, Evangelisch) und Experte für Buddhismus.
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

    try:
        client = get_llm_client()

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        analysis_json = json.loads(content)

        if 'relevant_passages' in analysis_json and isinstance(analysis_json['relevant_passages'], list):
            print(f"LLM-Analyse erfolgreich: {len(analysis_json['relevant_passages'])} relevante Passagen gefunden.")
            return analysis_json['relevant_passages']
        else:
            print("FEHLER: LLM gab kein erwartetes 'relevant_passages' Array zurück.")
            return []

    except openai.APIConnectionError as e:
        print(f"FEHLER: Verbindung zur Unsloth-API ({UNSLOTH_BASE_URL}) nicht möglich. Ist der Server gestartet?\n{e}")
        return []
    except openai.AuthenticationError:
        print("FEHLER: Authentifizierung bei der API fehlgeschlagen.")
        return []
    except json.JSONDecodeError:
        print("FEHLER: LLM gab ungültiges JSON zurück.")
        return []
    except Exception as e:
        print(f"FEHLER während der LLM-Analyse: {e}")
        return []


# ==============================================================================
# 3. ANZEIGE & BESTÄTIGUNG
# ==============================================================================

def display_and_get_confirmation(segments: list[dict], relevant_passages: list[dict], theme: str) -> bool:
    """Zeigt das transkribierte Material an und fragt nach Zustimmung."""
    print("\n" + "=" * 50)
    print(f"✅ Transkript und Analyse für Thema: '{theme}'")
    print("=" * 50)

    # Vollständiges Transkript
    print("\n--- VOLLSTÄNDIGES TRANSKRIPT (mit Zeitcodes) ---")
    for seg in segments:
        start_t = timedelta(seconds=seg['start'])
        end_t = timedelta(seconds=seg['end'])
        print(f"[{start_t.seconds // 60:02d}:{start_t.seconds % 60:02d} - {end_t.seconds // 60:02d}:{end_t.seconds % 60:02d}]: {seg['text']}")

    # Markierte relevante Passagen
    print("\n--- RELEVANTE PASSAGEN (markiert) ---")
    for rp in relevant_passages:
        start_t = timedelta(seconds=rp['start'])
        end_t = timedelta(seconds=rp['end'])
        print(f"👤 [Relevanz: {rp['relevance_score']}/10] 🎙️ [{start_t.seconds // 60:02d}:{start_t.seconds % 60:02d} - {end_t.seconds // 60:02d}:{end_t.seconds % 60:02d}] - {rp['text'][:80]}...")
        print(f"   -> Zusammenfassung: {rp['summary_german']}")

    print("\n" + "=" * 50)

    # Bestätigung abfragen
    while True:
        confirmation = input("Zustimmen Sie dem Schnitt dieser Passagen? (j/n/edit): ").lower()
        if confirmation == 'j':
            return True
        elif confirmation == 'n':
            feedback = input("Was soll verbessert werden? (z.B. 'Zu wenig Fokus auf Karma' oder 'Passagen 3 & 5 sind zu lang'): ")
            print(f"\n-> Feedback erhalten: {feedback}")
            return False
        elif confirmation == 'edit':
            print("\n--- Bearbeitungsmodus: Passagen manuell anpassen/entfernen ---")
            print("Passagen manuell angepasst (Simulation). Fortfahren mit aktueller Liste.")
            return True
        else:
            print("Ungültige Eingabe. Bitte 'j', 'n' oder 'edit' eingeben.")


# ==============================================================================
# 4. AUDIO-SCHNITT
# ==============================================================================

def trim_audio(audio: AudioSegment, relevant_passages: list[dict]) -> AudioSegment:
    """Schneidet den Audio-Track basierend auf den relevanten Timecodes."""
    print("\n--- Starte Audio-Zuschneiden ---")

    sorted_passages = sorted(relevant_passages, key=lambda x: x['start'])

    if not sorted_passages:
        print("FEHLER: Keine Passagen zum Schneiden gefunden.")
        return audio

    start_ms = int(sorted_passages[0]['start'] * 1000)
    end_ms = int(sorted_passages[-1]['end'] * 1000)

    print(f"Zuschneiden von {start_ms / 1000:.2f}s bis {end_ms / 1000:.2f}s...")

    trimmed_clip = audio[start_ms:end_ms]
    trimmed_clip.export(AUDIO_OUTPUT_TRIMMED_PATH, format="mp3")
    print(f"✅ Erfolg! Geschnittener Audio-Track gespeichert unter: {AUDIO_OUTPUT_TRIMMED_PATH}")

    return trimmed_clip


# ==============================================================================
# HAUPT-WORKFLOW
# ==============================================================================

def main_workflow():
    """Führt den gesamten Prozess durch."""
    print("=====================================================")
    print("🚀 STARTE AUDIO-TRANSKRIPTIONS- & ANALYSE-WORKFLOW")
    print(f"🎯 Thema: {THEME_TO_SEARCH}")
    print(f"🤖 LLM-Modus: {'Lokal (Unsloth)' if USE_LOCAL_UNSLOTH else 'OpenAI Cloud'}")
    print("=====================================================")

    # 1. Audio laden
    audio = load_audio(AUDIO_INPUT_PATH)
    if audio is None:
        return

    # 2. Transkribieren (lokal mit faster-whisper)
    segments = transcribe_audio(audio)
    if not segments:
        print("Workflow beendet: Keine Segmente transkribiert.")
        return

    # 3. LLM-Analyse (über Unsloth oder OpenAI)
    relevant_passages = analyze_transcript(segments, THEME_TO_SEARCH)
    if not relevant_passages:
        print("Workflow beendet: Keine relevanten Passagen gefunden.")
        return

    # 4. Anzeigen und Bestätigen
    user_agreed = display_and_get_confirmation(segments, relevant_passages, THEME_TO_SEARCH)

    # 5. Schnitt oder Wiederholung
    if user_agreed:
        trimmed_audio = trim_audio(audio, relevant_passages)

        print("\n" + "#" * 60)
        print("✨ WORKFLOW ERFOLGREICH ABGESCHLOSSEN ✨")
        print("#" * 60)
        print(f"▶️ Original Audio: {AUDIO_INPUT_PATH}")
        print(f"▶️ Geschnittener Audio: {AUDIO_OUTPUT_TRIMMED_PATH}")
        print(f"▶️ Anzahl relevanter Passagen: {len(relevant_passages)}")
        print(f"▶️ Thematischer Fokus: {THEME_TO_SEARCH}")
        print("\n--- Details zu den Passagen ---")
        for i, rp in enumerate(relevant_passages):
            start_t = timedelta(seconds=rp['start'])
            end_t = timedelta(seconds=rp['end'])
            print(f"  {i+1}. [{start_t.seconds // 60:02d}:{start_t.seconds % 60:02d} - {end_t.seconds // 60:02d}:{end_t.seconds % 60:02d}]")
            print(f"     Text: {rp['text'][:100]}...")
            print(f"     Zusammenfassung: {rp['summary_german']}")
    else:
        print("\n" + "!" * 60)
        print("🔄 WIEDERHOLUNG: Führe Analyse mit eventuellem Feedback durch.")
        print("!" * 60)
        main_workflow()


if __name__ == "__main__":
    main_workflow()