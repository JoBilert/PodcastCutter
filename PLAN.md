# Projektplan: PodcastCutter

**Ziel:** Audio-Datei (z.B. `input_dialog.mp3`) in ein strukturiertes Transkript umwandeln, relevante LLM-Passagen markieren, auf Benutzereingabe warten und gegebenenfalls den Audio-Track zuschneiden.

**Umgesetzte Schritte:**
1. ✅ **Transkription:** Lokal mit `openai-whisper` (offizielles OpenAI-Modell, läuft lokal, nutzt ffmpeg statt PyAV).
2. ✅ **LLM-Analyse:** Über OpenAI Cloud (Standard) oder lokale Unsloth/vLLM-API, umschaltbar via GUI. API-Key für Cloud erforderlich.
3. ✅ **GUI:** Gradio-Interface für Datei-Upload, Einstellungen, Ergebnisanzeige und Bestätigung.
4. ✅ **Feedback:** Benutzerzustimmung einholen (`✅` / `❌`), bei Ablehnung wiederholen mit Feedback.
5. ✅ **Schnitt:** Audio nach Zeitcodes zuschneiden und als `output_trimmed.mp3` speichern.

**Nächste Schritte:**
*   [ ] End-to-End-Test mit echter `input_dialog.mp3` und gestarteter lokaler Unsloth-API.
*   [ ] Fehlerbehandlung bei fehlender Unsloth-API-Verbindung.
*   [ ] Iterationszähler begrenzen, um Endlosschleifen zu vermeiden.
*   [ ] Mehrere separate Clips pro Passage statt einem Gesamt-Schnitt.