# Vocal Riffs and Runs Coach

A desktop application for vocal pitch detection and analysis from audio files, designed to help singers learn vocal runs and riffs for karaoke practice.

## Features

- Load audio files (MP3, WAV, FLAC)
- Real-time pitch detection and note identification
- Visual display of detected notes with timing
- Audio playback controls (play/pause, seek)
- Note visualization with note names (C4, D#5, etc.)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

## Requirements

- Python 3.8+
- See requirements.txt for Python package dependencies

## Future Features

- Vocal isolation using source separation
- Loop specific sections
- Speed control (slow down without pitch change)
- Export detected notes to MIDI or text format
- Real-time microphone input
