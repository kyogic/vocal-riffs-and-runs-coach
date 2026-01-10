# Vocal Riffs and Runs Coach

A desktop application for vocal pitch detection and analysis from audio files, designed to help singers learn vocal runs and riffs for karaoke practice.

## Features

### Audio Sources
- 📁 Load local audio files (MP3, WAV, FLAC)
- 🎵 Download audio directly from YouTube
- Support for various audio formats

### Analysis & Visualization
- Real-time pitch detection and note identification
- Visual display of detected notes with timing
- Note visualization with note names (C4, D#5, etc.)
- Waveform display

### Playback
- Audio playback controls (play/pause, seek, stop)
- Smooth, stutter-free playback
- Progress tracking with visual indicator

### Export
- Export detected notes to text or CSV format
- Save analysis for future reference

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

3. **(Optional) Install FFmpeg for YouTube support:**

**Windows (using Chocolatey):**
```bash
choco install ffmpeg
```

**Windows (manual):**
- Download from: https://www.ffmpeg.org/download.html
- Extract and add to system PATH

## Usage

Run the application:
```bash
python main.py
```

### Using YouTube

1. Copy a YouTube URL
2. Paste into the "Or YouTube URL:" field
3. Click "🎵 Download from YouTube"
4. Wait for download and automatic loading
5. Click "🔍 Analyze Pitch" to detect notes

See [YOUTUBE_SPOTIFY.md](YOUTUBE_SPOTIFY.md) for detailed YouTube integration guide.

## Requirements

- Python 3.8+
- FFmpeg (optional, for YouTube downloads and MP3 support)
- See requirements.txt for Python package dependencies

## Future Features

- Vocal isolation using source separation
- Loop specific sections
- Speed control (slow down without pitch change)
- Export detected notes to MIDI or text format
- Real-time microphone input
