# Audio Playback Troubleshooting Guide

If you're having issues with audio playback in the Vocal Coach application, follow this guide.

## Quick Diagnostic Test

Run the audio test script to diagnose issues:

```powershell
python test_audio.py
```

This will:
- ✅ Check if all packages are installed
- ✅ Detect available audio devices
- ✅ Test audio playback with a beep
- ✅ Test audio file processing and pitch detection

## Common Issues and Solutions

### Issue 1: "sounddevice not available" Error

**Symptoms:**
- Error message when clicking Play button
- Message: "Audio playback not available. Please install sounddevice"

**Solution:**
```powershell
# Activate your virtual environment first
.\venv\Scripts\Activate.ps1

# Install sounddevice
pip install sounddevice

# Or reinstall if already installed
pip install --force-reinstall sounddevice
```

### Issue 2: "PortAudio library not found"

**Symptoms:**
- ImportError when trying to import sounddevice
- Message about missing PortAudio library

**Solution for Windows:**

PortAudio should come bundled with sounddevice on Windows, but if it doesn't:

1. **Option A: Install pre-built wheel**
   ```powershell
   pip install --upgrade sounddevice --prefer-binary
   ```

2. **Option B: Install from conda (if using Anaconda)**
   ```powershell
   conda install -c conda-forge python-sounddevice
   ```

3. **Option C: Manual PortAudio installation**
   - Download PortAudio from: http://www.portaudio.com/download.html
   - Extract and place DLLs in your Python directory or system PATH

### Issue 3: No Sound but No Errors

**Symptoms:**
- Play button works
- No error messages
- But you don't hear anything

**Solutions:**

1. **Check system volume**
   - Make sure Windows volume is not muted
   - Check application volume in Volume Mixer

2. **Check default audio device**
   ```powershell
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```
   This will list all audio devices. Make sure your speakers/headphones are the default.

3. **Select audio device manually**

   You can set the default device in Python:
   ```python
   import sounddevice as sd
   sd.default.device = 1  # Try different numbers
   ```

### Issue 4: Audio Stuttering or Choppy Playback

**Symptoms:**
- Audio plays but with interruptions
- Crackling or stuttering sound

**Solutions:**

1. **Increase buffer size**

   Modify `main.py` to increase the audio buffer (advanced):
   ```python
   sd.default.blocksize = 2048  # Increase from default
   ```

2. **Close other audio applications**
   - Close Spotify, YouTube, etc.
   - Only one application should use audio at a time

3. **Update audio drivers**
   - Update your sound card drivers from Device Manager
   - Or download latest drivers from manufacturer's website

### Issue 5: "Permission Denied" or "Access Denied"

**Symptoms:**
- Cannot access audio device
- Permission errors

**Solutions:**

1. **Run as Administrator** (not recommended long-term)
   - Right-click PowerShell → Run as Administrator
   - Navigate to project and run application

2. **Check exclusive mode**
   - Right-click speaker icon → Sounds → Playback tab
   - Select your device → Properties → Advanced
   - Uncheck "Allow applications to take exclusive control"

### Issue 6: Different Audio Formats Not Supported

**Symptoms:**
- MP3 files won't load
- Error loading certain audio formats

**Solutions:**

1. **Install FFmpeg** (required for MP3 and other formats)

   **Windows:**
   - Download from: https://www.ffmpeg.org/download.html
   - Extract and add to PATH

   Or use Chocolatey:
   ```powershell
   choco install ffmpeg
   ```

2. **Install audioread** (librosa dependency for MP3)
   ```powershell
   pip install audioread
   ```

3. **Convert to WAV format**
   - Use an online converter or Audacity
   - WAV is the most compatible format

## Testing Individual Components

### Test 1: Check if sounddevice is installed

```powershell
python -c "import sounddevice; print('sounddevice version:', sounddevice.__version__)"
```

Expected output: Version number (e.g., `0.4.6`)

### Test 2: List audio devices

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Expected output: List of audio devices with indices

### Test 3: Test basic playback

```powershell
python -c "import sounddevice as sd; import numpy as np; sd.play(np.random.randn(44100), 44100); sd.wait()"
```

Expected output: 1 second of white noise

### Test 4: Test librosa

```powershell
python -c "import librosa; print('librosa version:', librosa.__version__)"
```

Expected output: Version number (e.g., `0.10.1`)

## Advanced Troubleshooting

### Enable Debug Mode

Add this to the top of `main.py` to see more error details:

```python
import sounddevice as sd
sd.default.extra_settings = sd.CoreAudioSettings(channel_map=[0])
```

### Check Audio Backend

```powershell
python -c "import sounddevice as sd; print('Backend:', sd.get_portaudio_version())"
```

### Alternative: Use Different Audio Backend

If sounddevice doesn't work, you can try pygame for audio:

```powershell
pip install pygame
```

Then modify the AudioPlayer class to use pygame instead (this would require code changes).

## Still Having Issues?

If none of these solutions work:

1. **Create a detailed bug report:**
   - Run `python test_audio.py` and save the output
   - Note your Python version: `python --version`
   - Note your OS version
   - List what you've tried

2. **Try a minimal test:**
   ```powershell
   # Create test.py
   import sounddevice as sd
   import numpy as np

   # Generate 440 Hz sine wave
   duration = 1  # second
   sample_rate = 44100
   t = np.linspace(0, duration, int(sample_rate * duration))
   audio = 0.5 * np.sin(2 * np.pi * 440 * t)

   print("Playing tone...")
   sd.play(audio, sample_rate)
   sd.wait()
   print("Done!")
   ```

   If this doesn't work, the issue is with sounddevice/PortAudio, not the Vocal Coach app.

3. **Check system requirements:**
   - Windows 10 or later (recommended)
   - Python 3.8 or higher
   - Working audio output device
   - Administrator privileges may be needed for first install

## Working Configuration Examples

Here are confirmed working configurations:

**Windows 10/11:**
- Python 3.10.x
- sounddevice 0.4.6
- numpy 1.24.3
- PyQt5 5.15.9

**Windows 10:**
- Python 3.9.x
- sounddevice 0.4.5
- numpy 1.23.x
- PyQt5 5.15.7

If your configuration matches one of these, everything should work!
