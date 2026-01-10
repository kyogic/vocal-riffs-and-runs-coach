# Usage Guide

## Quick Start

1. **Launch the application**
   ```bash
   python main.py
   ```

2. **Load an audio file**
   - Click "Load Audio File" button
   - Select an MP3, WAV, or FLAC file
   - The waveform will display automatically

3. **Analyze the pitch**
   - Click "🔍 Analyze Pitch" button
   - Wait for the analysis to complete
   - Detected notes will appear as colored regions with labels

4. **Playback controls**
   - Click "▶ Play" to start playback
   - Click "⏸ Pause" to pause
   - Click "⏹ Stop" to stop and reset
   - Drag the progress bar to seek through the audio
   - Red line shows current playback position

5. **Export detected notes**
   - Click "💾 Export Notes" button
   - Choose a location and format (.txt or .csv)
   - Notes will be saved with timing and pitch information

## Features Explained

### Pitch Detection
The application uses the PYIN algorithm (a variant of YIN) for robust pitch detection. It:
- Detects frequencies between C2 (65 Hz) and C7 (2093 Hz) - the typical vocal range
- Identifies note names (e.g., C4, D#5, A3)
- Calculates the duration of each note
- Handles pitch variations and vibrato

### Visualization
- **Top plot**: Audio waveform showing amplitude over time
- **Bottom plot**: Detected pitch frequency with colored note regions
- **Red vertical line**: Current playback position
- **Note labels**: Show the musical note name (e.g., C4, G#5)

### Export Formats

#### Text Format (.txt)
Human-readable format with note details:
```
Note 1:
  Time: 1.234 s
  Note: C4
  Frequency: 261.63 Hz
  Duration: 0.456 s
```

#### CSV Format (.csv)
Spreadsheet-compatible format:
```
Time (s),Note,Frequency (Hz),Duration (s)
1.234,C4,261.63,0.456
```

## Tips for Best Results

1. **Use clean vocal recordings**: The pitch detector works best with clear, isolated vocal tracks

2. **Start with simple melodies**: Begin with songs that have clear, distinct notes before analyzing complex riffs

3. **Export for practice**: Export the detected notes to review the melody structure and practice specific sections

4. **Audio quality matters**: Higher quality audio files (WAV, FLAC) typically give better results than compressed MP3s

## Troubleshooting

### No audio playback
- Make sure `sounddevice` is installed: `pip install sounddevice`
- Check your system audio settings
- Try a different audio file

### Inaccurate pitch detection
- Ensure the audio contains clear vocals
- Try using a higher quality audio file
- Vocal isolation (planned feature) will help with music tracks

### Application crashes
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Make sure you have Python 3.8 or higher
- Check the console for error messages

## Keyboard Shortcuts (Future Feature)
- Space: Play/Pause
- Left/Right arrows: Seek backward/forward
- R: Restart from beginning
- E: Export notes

## Next Steps

After getting comfortable with the basic features, you can:
1. Practice singing along with the detected notes
2. Export notes to compare different performances
3. Use the timing information to practice difficult sections
4. Look forward to upcoming features like vocal isolation and speed control!
