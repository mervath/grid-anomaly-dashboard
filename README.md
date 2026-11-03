# Grid Anomaly Dashboard

A Tkinter desktop app for signal anomaly reconstruction and a voice assistant that can summarize results through Together AI. This repository is a compact application cut from the supplied archive: it uses the self-contained `complete_dashboard.py` version as `main.py` and the matching small model checkpoint. Thousands of generated images, alternate code versions, databases, logs, audio/video, presentations, and API-key files were omitted.

## Requirements

- Windows with Python 3.10 or 3.11 recommended
- A microphone and speakers for voice features (optional)
- The included `signal2signal_cgan.pth` checkpoint
- A Together AI API key for voice/LLM features

## Setup and run (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set your own TOGETHER_API_KEY value
python main.py
```

Tkinter is normally included with the Windows Python installer. `faster-whisper` may download its speech model the first time voice transcription is used. The `.env` file is ignored by Git and must never be committed.

## Model and data files

The matching `signal2signal_cgan.pth` model checkpoint is included. The other model weights, generated images, videos, logs, and databases from the archive are not required by this selected app version and were left out. The app creates local outputs as it runs; review those outputs before sharing them.

## License

Add a license only after confirming you own the source and model or have permission to publish them.
