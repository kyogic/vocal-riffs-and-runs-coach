# Sound Library Integrations for Better Note Playback

Your app currently uses simple sine wave synthesis. Here are much better options:

---

## Option 1: FluidSynth + SoundFonts (RECOMMENDED)

**Best for:** Realistic piano/instrument sounds

### What is FluidSynth?
- Software synthesizer that plays SoundFont (.sf2) files
- Professional-quality instrument sounds
- Used in many professional music apps
- Low latency, high quality

### Installation:

```bash
# Install FluidSynth library (system)
sudo apt-get install fluidsynth libfluidsynth-dev  # Linux
brew install fluid-synth  # macOS
# Windows: Download from https://github.com/FluidSynth/fluidsynth/releases

# Install Python bindings
pip install pyfluidsynth
```

### Get a Free SoundFont:

```bash
# Download a free, high-quality piano SoundFont
wget https://github.com/FluidSynth/fluidsynth/raw/master/sf2/VintageDreamsWaves-v2.sf2

# Or use MuseScore's SoundFont (excellent quality)
wget https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3
```

### Integration Code:

```python
import pyfluidsynth
import numpy as np

class FluidSynthPlayer:
    """High-quality note playback using FluidSynth"""

    def __init__(self, soundfont_path='MuseScore_General.sf3'):
        self.fs = pyfluidsynth.Synth()
        self.fs.start(driver='alsa')  # Use 'coreaudio' on macOS, 'dsound' on Windows

        # Load SoundFont
        self.sfid = self.fs.sfload(soundfont_path)
        self.fs.program_select(0, self.sfid, 0, 0)  # Channel 0, Piano

    def play_note(self, note_midi, duration, velocity=80):
        """
        Play a note with realistic piano sound

        Args:
            note_midi: MIDI note number (60 = middle C)
            duration: Duration in seconds
            velocity: Volume (0-127)
        """
        # Note on
        self.fs.noteon(0, note_midi, velocity)

        # Wait for duration
        time.sleep(duration)

        # Note off
        self.fs.noteoff(0, note_midi)

    def play_notes_sequence(self, notes):
        """Play a sequence of notes"""
        for note in notes:
            freq = note['frequency']
            duration = max(note['duration'], 0.2)  # Minimum 200ms

            # Convert frequency to MIDI note
            midi_note = int(round(librosa.hz_to_midi(freq)))

            # Play note
            self.play_note(midi_note, duration)

            # Small gap between notes
            time.sleep(0.05)

    def cleanup(self):
        """Clean up resources"""
        self.fs.delete()
```

### Benefits:
✓ Realistic piano sound
✓ Professional quality
✓ Low latency
✓ Free SoundFonts available
✓ Widely used in music software

---

## Option 2: pretty_midi + pygame (Easier)

**Best for:** Quick integration without system dependencies

### Installation:

```bash
pip install pretty_midi pygame
```

### Integration Code:

```python
import pretty_midi
import pygame
import tempfile
import os

class MIDIPlayer:
    """Convert notes to MIDI and play with pygame"""

    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    def play_notes_sequence(self, notes):
        """Play notes as MIDI"""
        # Create MIDI file
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        current_time = 0.0
        for note in notes:
            freq = note['frequency']
            duration = max(note['duration'], 0.2)

            # Convert to MIDI note
            midi_note = int(round(librosa.hz_to_midi(freq)))

            # Create note
            midi_note_obj = pretty_midi.Note(
                velocity=100,
                pitch=midi_note,
                start=current_time,
                end=current_time + duration
            )
            piano.notes.append(midi_note_obj)

            current_time += duration + 0.05  # Gap

        midi.instruments.append(piano)

        # Save to temp file and play
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            midi.write(f.name)
            temp_file = f.name

        # Play with pygame
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        # Wait for playback
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Cleanup
        os.unlink(temp_file)
```

### Benefits:
✓ Easy to install (no system dependencies)
✓ Good quality MIDI playback
✓ Cross-platform
✓ pygame handles audio automatically

### Drawbacks:
⚠ Requires MIDI file generation
⚠ Quality depends on system MIDI synth

---

## Option 3: Pyo (Advanced DSP)

**Best for:** Custom sound design, advanced effects

### Installation:

```bash
pip install pyo
```

### Integration Code:

```python
from pyo import *

class PyoPlayer:
    """Advanced synthesis with Pyo"""

    def __init__(self):
        self.server = Server().boot()
        self.server.start()

    def play_note(self, frequency, duration):
        """Play a note with rich harmonics"""
        # Create oscillators (richer sound than pure sine)
        osc1 = Sine(freq=frequency, mul=0.3)
        osc2 = Sine(freq=frequency*2, mul=0.15)  # Octave
        osc3 = Sine(freq=frequency*3, mul=0.1)   # Fifth

        # Add envelope for natural attack/release
        env = Adsr(attack=0.01, decay=0.1, sustain=0.7, release=0.3, dur=duration)

        # Mix oscillators and apply envelope
        sound = Mix([osc1, osc2, osc3], voices=1)
        output = sound * env
        output.out()

        # Trigger envelope
        env.play()

        # Wait for note to finish
        time.sleep(duration)
```

### Benefits:
✓ Very powerful DSP capabilities
✓ Can create custom timbres
✓ Real-time effects (reverb, chorus, etc.)
✓ Professional audio quality

### Drawbacks:
⚠ Complex API
⚠ Overkill for simple playback

---

## Option 4: Karplus-Strong (Plucked String)

**Best for:** Guitar/string-like sounds

### Already have numpy/scipy? Add this:

```python
import numpy as np
from scipy import signal

def karplus_strong_note(frequency, duration, sr=44100, pluck_position=0.5):
    """
    Generate plucked string sound using Karplus-Strong algorithm
    Sounds like a guitar/harp
    """
    # Calculate delay length
    delay_samples = int(sr / frequency)

    # Initialize delay line with noise (the "pluck")
    delay_line = np.random.uniform(-1, 1, delay_samples)

    # Generate samples
    num_samples = int(duration * sr)
    output = np.zeros(num_samples)

    for i in range(num_samples):
        # Output current sample
        output[i] = delay_line[0]

        # Apply lowpass filter (averaging)
        new_sample = 0.996 * (delay_line[0] + delay_line[1]) / 2

        # Shift delay line
        delay_line[:-1] = delay_line[1:]
        delay_line[-1] = new_sample

    # Apply envelope
    envelope = np.exp(-3 * np.linspace(0, 1, num_samples))
    output *= envelope

    return output

# Use in your AudioSynthPlayer:
def generate_note_audio(self, frequency, duration, sample_rate=44100):
    # Replace sine wave generation with Karplus-Strong
    return karplus_strong_note(frequency, duration, sample_rate)
```

### Benefits:
✓ No new dependencies
✓ Better sound than sine waves
✓ Guitar/harp-like timbre
✓ Easy to integrate

---

## Comparison Table:

| Option | Quality | Setup | Dependencies | Cross-Platform | Best For |
|--------|---------|-------|--------------|----------------|----------|
| **FluidSynth** | ⭐⭐⭐⭐⭐ | Medium | System + Python | ✓ | **Realistic piano** |
| pretty_midi + pygame | ⭐⭐⭐⭐ | Easy | Python only | ✓ | Quick improvement |
| Pyo | ⭐⭐⭐⭐⭐ | Hard | Python only | ✓ | Custom sounds |
| Karplus-Strong | ⭐⭐⭐ | Very Easy | None | ✓ | **Quick upgrade** |
| Current (sine waves) | ⭐⭐ | Already done | None | ✓ | Testing |

---

## Recommendation:

### **Easiest Upgrade** (5 minutes):
Use **Karplus-Strong** - Replace your sine wave generation with the plucked string algorithm. No new dependencies, sounds much better.

### **Best Quality** (30 minutes):
Use **FluidSynth + SoundFont** - Professional piano sound, worth the setup time.

### **Good Balance** (15 minutes):
Use **pretty_midi + pygame** - Good quality, easy setup, no system dependencies.

---

## Quick Implementation: Karplus-Strong

Let me integrate Karplus-Strong into your existing code since it requires no new dependencies:

```python
# In AudioSynthPlayer class, replace generate_note_audio method:

def generate_note_audio(self, frequency, duration, sample_rate=44100):
    """Generate plucked string sound using Karplus-Strong algorithm"""
    # Calculate delay length
    delay_samples = int(sample_rate / frequency)

    # Initialize delay line with noise burst (the "pluck")
    delay_line = np.random.uniform(-0.5, 0.5, delay_samples)

    # Generate samples
    num_samples = int(duration * sample_rate)
    output = np.zeros(num_samples)

    for i in range(num_samples):
        # Output current sample
        output[i] = delay_line[0]

        # Apply lowpass filter for decay (averaging)
        new_sample = 0.996 * (delay_line[0] + delay_line[1]) / 2

        # Shift delay line and add feedback
        delay_line = np.roll(delay_line, -1)
        delay_line[-1] = new_sample

    # Apply overall envelope for more natural sound
    attack_samples = int(0.01 * sample_rate)
    release_samples = int(0.1 * sample_rate)

    # Attack
    if num_samples > attack_samples:
        attack_env = np.linspace(0, 1, attack_samples)
        output[:attack_samples] *= attack_env

    # Release
    if num_samples > release_samples:
        release_env = np.linspace(1, 0, release_samples)
        output[-release_samples:] *= release_env

    # Normalize
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val

    # Reduce volume to prevent clipping
    output *= 0.6

    return output.astype(np.float32)
```

This sounds much better than sine waves and requires no new libraries!

---

## Next Steps:

1. **Try Karplus-Strong first** (easiest upgrade)
2. **If you want piano sound**, install FluidSynth
3. **For advanced customization**, explore Pyo

Would you like me to integrate any of these into your app?
