# PodcastCutter

A Python program for automatic analysis of audio dialogs with timecode transcription, LLM-based topic analysis and user-controlled audio cutting.

All processing steps run locally (transcription via openai-whisper). The LLM analysis runs over any OpenAI-compatible API.

## Quick Start (Windows)

1. Download this repository (or clone it).
2. Double-click **`start.bat`**.
3. The script does everything else:
   * Checks and installs Python 3.12 if needed
   * Checks and installs ffmpeg if needed
   * Creates a virtual Python environment (`.venv`)
   * Installs all Python dependencies (`openai-whisper`, `pydub`, `gradio`)
   * Starts the GUI
   * Opens Microsoft Edge at `http://localhost:7860`

## Requirements

*   **Windows 10 or 11** (for `start.bat` and winget)
*   **Administrator rights** (if Python or ffmpeg need to be installed)
*   **Internet connection** (for the first start to install Python/ffmpeg/packages)
*   **NVIDIA NIM API Key** (or OpenAI/OpenRouter/Unsloth) – enter at first start in Tab 0

## What Happens on First Start?

1. `start.bat` checks if Python and ffmpeg are installed. If not, they are installed automatically via `winget`.
2. A virtual environment under `.venv` is created.
3. `pip install -r requirements.txt` installs all Python packages.
4. The Gradio GUI starts.
5. Microsoft Edge opens automatically with `http://localhost:7860`.

## API Setup (Tab 0)

On first start of the GUI, tab **`0. 🔧 Setup`** opens automatically:

1. Select an API provider (NVIDIA NIM, OpenAI, OpenRouter, Unsloth/vLLM or custom API).
2. Enter API Key, Base URL and Model name.
3. Click "Save Configuration".
4. Switch to Tab 1 and upload your podcast file.

### NVIDIA NIM Access

1. Go to https://build.nvidia.com/nim and create an account.
2. Search for an LLM model (e.g. `nemotron-3.5-super-vl-120b-a12b`).
3. Create an API Key under "API Keys" in the dashboard.

### Local Unsloth/vLLM

1. Start a local server, e.g.:
   ```bash
   python -m unsloth.vllm.server --model unsloth/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8888
   ```
2. Base URL: `http://127.0.0.1:8888/v1`
3. API key can be left empty or set to "not-needed".

## Manual Installation (without start.bat)

```bash
# Enter project directory
cd /home/talmar/HermesProjekte/PodcastCutter

# Install dependencies
pip install -r requirements.txt

# Start GUI
python3 app.py
```

## Project Structure

```
PodcastCutter/
├── app.py               # Gradio GUI (main entry: python3 app.py)
├── audio_processor.py   # CLI workflow (alternative without GUI)
├── start.bat            # Self-contained Windows start (Python, ffmpeg, venv, Edge)
├── PLAN.md              # Project plan and next steps
├── README.md            # This documentation (German)
├── README_EN.md         # This documentation (English)
├── requirements.txt     # Python dependencies
├── input_dialog.mp3     # Placeholder audio (must be provided by user)
├── .gitignore           # Git ignore rules
└── .env                 # Configuration (created on first run, NOT committed to repo)
```

## Git Branches

*   `master` – Local working version (may contain `.env` with API key)
*   `clean-setup` – Clean version without API secrets, with first-run setup flow. `.env` is in `.gitignore` and is not committed.

Switch branches:
```bash
git checkout clean-setup   # Clean version without secrets
git checkout master        # Local version
```

## Credits & Licenses

This project uses the following libraries and systems:

| Component | License | Source |
|-----------|---------|--------|
| Python | [Python Software Foundation License](https://docs.python.org/3/license.html) | https://python.org |
| openai-whisper | [MIT License](https://github.com/openai/whisper/blob/main/LICENSE) | https://github.com/openai/whisper |
| pydub | [MIT License](https://github.com/jiaaro/pydub/blob/master/LICENSE) | https://github.com/jiaaro/pydub |
| gradio | [Apache License 2.0](https://github.com/gradio-app/gradio/blob/main/LICENSE) | https://github.com/gradio-app/gradio |
| ffmpeg | [GNU General Public License v2+](https://ffmpeg.org/legal.html) | https://ffmpeg.org |
| NVIDIA NIM API | NVIDIA Terms of Service (proprietary) | https://build.nvidia.com/nim |
| OpenAI API | OpenAI Terms of Service (proprietary) | https://openai.com/policies/terms-of-use |
| OpenRouter API | OpenRouter Terms of Service (proprietary) | https://openrouter.ai |

This project is under the MIT License. See [LICENSE](LICENSE) for details (if available).

**Note:** The use of cloud APIs (NVIDIA NIM, OpenAI, OpenRouter) is subject to the respective provider's terms of use and may incur costs. Local transcription via `openai-whisper` runs completely offline on your own hardware.