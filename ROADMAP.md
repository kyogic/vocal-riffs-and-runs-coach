# Roadmap & Future Features

## MVP Features ✅
- [x] Desktop GUI with PyQt5
- [x] Load audio files (MP3, WAV, FLAC)
- [x] Pitch detection using librosa
- [x] Visual display of detected notes
- [x] Note names and timing information
- [x] Basic playback controls (play/pause/seek)
- [x] Export detected notes to text/CSV

## Planned Features

### Phase 1: Enhanced Audio Control
- [ ] Loop specific sections
  - Select start and end points
  - Continuous loop playback
  - Quick loop markers

- [ ] Speed control without pitch change
  - Slow down to 50%, 75%, 90%
  - Use time-stretching algorithms (librosa or pyrubberband)
  - Maintain pitch accuracy

- [ ] Volume control
- [ ] Keyboard shortcuts for all controls

### Phase 2: Vocal Isolation
- [ ] Source separation using Spleeter or Demucs
  - Extract vocal track from music
  - Remove backing vocals (keep lead only)
  - Adjustable separation quality

- [ ] Pre-processing options
  - Noise reduction
  - Audio normalization
  - High-pass filter for vocals

### Phase 3: Advanced Analysis
- [ ] Pitch accuracy visualization
  - Show deviation from perfect pitch
  - Highlight sharp/flat notes
  - Display cents deviation

- [ ] Vibrato detection and analysis
  - Frequency and rate of vibrato
  - Visualize vibrato patterns

- [ ] Vocal runs detection
  - Identify melismatic passages
  - Highlight fast note transitions
  - Break down complex runs

- [ ] Compare with reference pitch
  - Load reference MIDI or audio
  - Overlay reference vs. detected
  - Show differences

### Phase 4: Practice Tools
- [ ] Metronome integration
  - Adjustable tempo
  - Visual and audio click

- [ ] Pitch guide overlay
  - Show target notes
  - Real-time comparison during playback

- [ ] Practice mode
  - Loop difficult sections automatically
  - Gradual speed increase
  - Track improvement over time

### Phase 5: Export & Sharing
- [ ] MIDI export
  - Convert detected notes to MIDI
  - Adjustable quantization

- [ ] Sheet music generation
  - Basic notation export
  - Integration with MuseScore or LilyPond

- [ ] Session saving
  - Save analysis results
  - Quick reload previous sessions

- [ ] Video export
  - Sync visualization with audio
  - Create practice videos

### Phase 6: Real-time Features
- [ ] Microphone input
  - Real-time pitch detection
  - Live feedback while singing

- [ ] Pitch correction suggestions
  - Show which notes were off
  - Recommend practice exercises

- [ ] Record and compare
  - Record your performance
  - Compare with original
  - Track progress over time

### Technical Improvements
- [ ] Performance optimization
  - Faster pitch detection
  - Cached analysis results
  - Multi-threading for heavy operations

- [ ] Better error handling
  - Graceful degradation
  - User-friendly error messages
  - Recovery from crashes

- [ ] Cross-platform packaging
  - Windows executable (.exe)
  - macOS app bundle (.app)
  - Linux AppImage

- [ ] Unit tests
  - Test pitch detection accuracy
  - Test audio loading
  - GUI component tests

- [ ] Configuration file
  - Save user preferences
  - Custom pitch range settings
  - Default export locations

## Community Features (Long-term)
- [ ] Song database
  - Share detected note patterns
  - Community-contributed analyses
  - Rating and reviews

- [ ] Challenge mode
  - Practice specific vocal techniques
  - Achievement system
  - Leaderboards

- [ ] Integration with music services
  - Spotify, YouTube, Apple Music
  - Direct download and analysis

## Research & Exploration
- [ ] Machine learning for better vocal isolation
- [ ] Genre-specific pitch detection tuning
- [ ] Automatic key detection
- [ ] Chord recognition (for backing tracks)

## Feedback Welcome!
This is a living document. If you have suggestions or would like to prioritize certain features, please open an issue on GitHub!
