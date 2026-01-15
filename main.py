#!/usr/bin/env python3
"""
Vocal Riffs and Runs Coach
A desktop application for vocal pitch detection and analysis
"""

import sys
import os
import time
import numpy as np
import librosa
from scipy.ndimage import median_filter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider,
                             QFileDialog, QProgressBar, QSpinBox,
                             QLineEdit, QListWidget, QListWidgetItem, QGroupBox,
                             QDoubleSpinBox, QMessageBox, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class YouTubeDownloader(QThread):
    """Background thread for downloading YouTube audio"""
    progress = pyqtSignal(str)  # Progress message
    finished = pyqtSignal(str)  # Downloaded file path
    error = pyqtSignal(str)     # Error message

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.output_path = None

    def run(self):
        """Download audio from YouTube"""
        try:
            import yt_dlp

            # Create downloads directory if it doesn't exist
            download_dir = os.path.join(os.path.expanduser('~'), '.vocal_coach', 'downloads')
            os.makedirs(download_dir, exist_ok=True)

            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'quiet': False,
                'no_warnings': False,
                'extract_flat': False,
            }

            self.progress.emit('Connecting to YouTube...')

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info
                self.progress.emit('Fetching video information...')
                info = ydl.extract_info(self.url, download=False)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)

                self.progress.emit(f'Downloading: {title} ({duration//60}:{duration%60:02d})...')

                # Download and convert
                ydl.download([self.url])

                # Find the downloaded file
                output_filename = ydl.prepare_filename(info)
                # Change extension to wav (due to postprocessor)
                base_name = os.path.splitext(output_filename)[0]
                self.output_path = base_name + '.wav'

                if os.path.exists(self.output_path):
                    self.progress.emit('Download complete!')
                    self.finished.emit(self.output_path)
                else:
                    self.error.emit('Download completed but file not found')

        except ImportError:
            self.error.emit('yt-dlp not installed. Please run:\npip install yt-dlp')
        except Exception as e:
            self.error.emit(f'Download error: {str(e)}')


class BPMDetectionThread(QThread):
    """Background thread for BPM detection"""
    progress = pyqtSignal(str)  # Progress message
    finished = pyqtSignal(float)  # BPM value
    error = pyqtSignal(str)  # Error message

    def __init__(self, audio, sr):
        super().__init__()
        self.audio = audio
        self.sr = sr

    def run(self):
        """Detect BPM in background"""
        try:
            self.progress.emit('Detecting tempo (BPM)...')
            tempo, _ = librosa.beat.beat_track(y=self.audio, sr=self.sr)
            bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            self.progress.emit(f'Detected tempo: {bpm:.1f} BPM')
            self.finished.emit(bpm)
        except Exception as e:
            self.error.emit(f'Could not detect BPM: {str(e)}')


class PitchAnalysisThread(QThread):
    """Background thread for pitch analysis"""
    progress = pyqtSignal(str)  # Progress message
    progress_percent = pyqtSignal(int)  # Progress percentage (0-100)
    finished = pyqtSignal(list, object, object)  # notes, times, f0
    error = pyqtSignal(str)  # Error message

    def __init__(self, audio, sr, voiced_threshold=0.75, energy_threshold=0.15,
                 isolate_vocals=True, start_time=None, end_time=None,
                 algorithm='pyin', min_note_duration=0.05):
        super().__init__()
        self.audio = audio
        self.sr = sr
        self.voiced_threshold = voiced_threshold
        self.energy_threshold = energy_threshold
        self.isolate_vocals = isolate_vocals
        self.start_time = start_time
        self.end_time = end_time
        self.algorithm = algorithm
        self.min_note_duration = min_note_duration

    def run(self):
        """Analyze pitch in background"""
        try:
            # Extract region if specified
            audio_to_analyze = self.audio
            if self.start_time is not None or self.end_time is not None:
                start_sample = int(self.start_time * self.sr) if self.start_time else 0
                end_sample = int(self.end_time * self.sr) if self.end_time else len(self.audio)
                audio_to_analyze = self.audio[start_sample:end_sample]
                self.progress.emit(f'Analyzing region {self.start_time:.1f}s - {self.end_time:.1f}s')

            if self.isolate_vocals:
                self.progress.emit('Isolating vocals from background music...')
            else:
                self.progress.emit('Analyzing audio without vocal isolation...')
            self.progress_percent.emit(10)

            pitch_detector = PitchDetector(
                self.sr,
                voiced_threshold=self.voiced_threshold,
                energy_threshold=self.energy_threshold,
                algorithm=self.algorithm,
                min_note_duration=self.min_note_duration
            )

            self.progress.emit(f'Detecting melody using {self.algorithm.upper()} algorithm...')
            self.progress_percent.emit(40)
            notes, times, f0 = pitch_detector.get_note_data(
                audio_to_analyze,
                isolate_vocals=self.isolate_vocals
            )

            # Adjust note times if analyzing a region
            if self.start_time is not None and self.start_time > 0:
                for note in notes:
                    note['time'] += self.start_time
                times = times + self.start_time

            self.progress.emit(f'Analysis complete! Detected {len(notes)} notes')
            self.progress_percent.emit(100)
            self.finished.emit(notes, times, f0)

        except Exception as e:
            self.error.emit(f'Analysis error: {str(e)}')


class PitchDetector:
    """Handles pitch detection from audio data"""

    def __init__(self, sr=22050, voiced_threshold=0.75, energy_threshold=0.15,
                 algorithm='pyin', min_note_duration=0.05):
        self.sr = sr
        self.hop_length = 256  # Good time resolution for vocal runs
        # Typical vocal range: E2 (82 Hz) to E6 (1319 Hz)
        # Using slightly wider range to avoid cutting off edge notes
        self.fmin = librosa.note_to_hz('D2')  # ~73 Hz - below typical male vocals
        self.fmax = librosa.note_to_hz('A6')  # ~1760 Hz - above typical female vocals
        self.voiced_threshold = voiced_threshold
        self.energy_threshold = energy_threshold
        self.algorithm = algorithm
        self.min_note_duration = min_note_duration

    def isolate_vocals(self, audio):
        """
        Isolate vocals from background music using harmonic-percussive separation
        and spectral processing (less aggressive to preserve pitch accuracy)
        """
        # Use harmonic-percussive source separation with lower margin for gentler separation
        # Vocals are primarily harmonic
        harmonic, percussive = librosa.effects.hpss(audio, margin=2.0)

        # Less aggressive frequency filtering to preserve pitch accuracy
        S = librosa.stft(harmonic)

        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr)

        # Create a mask: keep frequencies in vocal range (60 Hz to 4000 Hz)
        # Wider range to avoid cutting off harmonics that help pitch detection
        freq_mask = (freqs >= 60) & (freqs <= 4000)

        # Apply mask to spectrogram - keep more of the original to preserve pitch
        S_filtered = S.copy()
        S_filtered[~freq_mask, :] = S_filtered[~freq_mask, :] * 0.3  # Less aggressive filtering

        # Convert back to audio
        vocals_isolated = librosa.istft(S_filtered)

        # Normalize carefully
        if np.max(np.abs(vocals_isolated)) > 0:
            vocals_isolated = vocals_isolated / np.max(np.abs(vocals_isolated))

        return vocals_isolated

    def detect_pitch(self, audio):
        """
        Detect pitch from audio using selected algorithm
        Returns frequencies, voiced flag, and voiced probabilities
        """
        if self.algorithm == 'yin':
            # YIN algorithm - simpler, sometimes more reliable for clean vocals
            # YIN doesn't return voiced_flag or voiced_probs, so we'll estimate them
            f0 = librosa.yin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=self.sr,
                hop_length=self.hop_length,
                frame_length=2048
            )
            # Estimate voiced flag (any non-NaN frequency is considered voiced)
            voiced_flag = ~np.isnan(f0)
            # For YIN, we'll create a simple confidence based on frequency stability
            voiced_probs = np.ones_like(f0)
            voiced_probs[np.isnan(f0)] = 0.0

            # Calculate confidence based on local stability
            for i in range(1, len(f0) - 1):
                if not np.isnan(f0[i]):
                    # Look at neighboring frames
                    neighbors = []
                    if not np.isnan(f0[i-1]):
                        neighbors.append(abs(f0[i] - f0[i-1]) / f0[i])
                    if i < len(f0) - 1 and not np.isnan(f0[i+1]):
                        neighbors.append(abs(f0[i] - f0[i+1]) / f0[i])

                    if neighbors:
                        # Stability = inverse of relative variation
                        variation = np.mean(neighbors)
                        voiced_probs[i] = max(0.0, 1.0 - variation * 10)

        elif self.algorithm == 'crepe':
            # CREPE algorithm - Deep learning based, very accurate but slower
            try:
                import crepe

                # CREPE expects 16kHz sample rate, resample if needed
                target_sr = 16000
                if self.sr != target_sr:
                    audio_resampled = librosa.resample(audio, orig_sr=self.sr, target_sr=target_sr)
                else:
                    audio_resampled = audio

                # Run CREPE (model_capacity: 'tiny', 'small', 'medium', 'large', 'full')
                # Using 'small' for balance between speed and accuracy
                time, frequency, confidence, activation = crepe.predict(
                    audio_resampled,
                    target_sr,
                    viterbi=True,  # Use Viterbi decoding for smoother pitch contours
                    model_capacity='small',  # Balance between speed and accuracy
                    step_size=10  # milliseconds between predictions
                )

                # Resample CREPE output to match our hop_length
                target_frames = int(len(audio) / self.hop_length)
                time_frames = np.arange(target_frames) * self.hop_length / self.sr

                # Interpolate frequency and confidence to match our frame rate
                f0 = np.interp(time_frames, time, frequency)
                voiced_probs = np.interp(time_frames, time, confidence)

                # Filter out frequencies outside our vocal range
                f0[(f0 < self.fmin) | (f0 > self.fmax)] = np.nan

                # Set unvoiced regions to NaN based on confidence threshold
                # CREPE's confidence is already quite good, so we use a lower threshold
                f0[voiced_probs < 0.3] = np.nan
                voiced_flag = ~np.isnan(f0)

            except ImportError:
                print("ERROR: CREPE not installed. Install with: pip install crepe tensorflow")
                print("Falling back to pYIN algorithm...")
                # Fall back to pYIN
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sr=self.sr,
                    hop_length=self.hop_length,
                    frame_length=2048,
                    win_length=1800,
                    fill_na=None
                )
            except Exception as e:
                print(f"ERROR running CREPE: {str(e)}")
                print("Falling back to pYIN algorithm...")
                # Fall back to pYIN
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sr=self.sr,
                    hop_length=self.hop_length,
                    frame_length=2048,
                    win_length=1800,
                    fill_na=None
                )

        else:  # pyin (default)
            # pYIN algorithm - probabilistic YIN with better handling of noise
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=self.sr,
                hop_length=self.hop_length,
                frame_length=2048,  # Good frequency resolution
                win_length=1800,    # Smaller window for better time resolution
                fill_na=None        # Keep NaN for unvoiced regions (don't interpolate)
            )

        return f0, voiced_flag, voiced_probs

    def frequency_to_note(self, frequency):
        """Convert frequency to note name"""
        if frequency is None or np.isnan(frequency):
            return None

        note_number = librosa.hz_to_midi(frequency)
        note_name = librosa.midi_to_note(int(round(note_number)))
        return note_name

    def median_filter_pitch(self, f0, window_size=5):
        """Apply median filter to remove pitch tracking errors"""
        # Create a copy to work with
        filtered_f0 = f0.copy()

        # Only filter non-NaN values
        valid_mask = ~np.isnan(f0)
        if np.sum(valid_mask) > window_size:
            # Get indices of valid values
            valid_indices = np.where(valid_mask)[0]
            valid_values = f0[valid_mask]

            # Apply median filter to valid values
            filtered_values = median_filter(valid_values, size=window_size, mode='nearest')

            # Put filtered values back
            filtered_f0[valid_indices] = filtered_values

        return filtered_f0

    def get_note_data(self, audio, isolate_vocals=True):
        """
        Analyze audio and return note data with timing
        Returns list of (time, frequency, note_name, duration)

        Args:
            audio: Audio signal
            isolate_vocals: If True, isolate vocals before pitch detection
        """
        # Isolate vocals if requested
        if isolate_vocals:
            audio = self.isolate_vocals(audio)

        f0, voiced_flag, voiced_probs = self.detect_pitch(audio)

        # Apply median filter to smooth out pitch tracking errors
        f0_filtered = self.median_filter_pitch(f0, window_size=5)

        # Convert frame indices to time
        times = librosa.frames_to_time(
            np.arange(len(f0_filtered)),
            sr=self.sr,
            hop_length=self.hop_length
        )

        # Calculate RMS energy for each frame to filter out silence
        # This prevents detecting "phantom notes" in quiet/silent sections
        frame_length = 2048
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=self.hop_length
        )[0]

        # Pad or trim rms to match f0 length
        if len(rms) < len(f0_filtered):
            rms = np.pad(rms, (0, len(f0_filtered) - len(rms)), mode='edge')
        elif len(rms) > len(f0_filtered):
            rms = rms[:len(f0_filtered)]

        # Calculate minimum energy threshold as a percentage of max energy
        # This adapts to the overall loudness of the audio
        max_energy = np.max(rms)
        min_energy_threshold = max_energy * self.energy_threshold  # Use configurable threshold

        notes = []
        current_note = None
        note_start = None
        freq_accumulator = []  # Accumulate frequencies for averaging

        # Use configurable voiced probability threshold (0.0 to 1.0)
        min_voiced_prob = self.voiced_threshold

        for i, (time, freq, voiced, prob, energy) in enumerate(zip(times, f0_filtered, voiced_flag, voiced_probs, rms)):
            # Only consider high-confidence voiced segments with sufficient energy
            # This filters out both uncertain detections AND silence/quiet sections
            if voiced and not np.isnan(freq) and prob >= min_voiced_prob and energy >= min_energy_threshold:
                note_name = self.frequency_to_note(freq)

                # Accumulate frequency for averaging
                freq_accumulator.append(freq)

                if note_name != current_note:
                    # Save previous note if exists
                    if current_note is not None and note_start is not None and len(freq_accumulator) > 0:
                        duration = time - note_start
                        # Use median frequency for robustness
                        avg_freq = np.median(freq_accumulator[:-1]) if len(freq_accumulator) > 1 else freq_accumulator[0]

                        # Only save notes with minimum duration (configurable, reduces noise)
                        if duration >= self.min_note_duration:
                            notes.append({
                                'time': note_start,
                                'frequency': avg_freq,
                                'note': current_note,
                                'duration': duration
                            })

                    # Start new note
                    current_note = note_name
                    note_start = time
                    freq_accumulator = [freq]
            else:
                # End of voiced segment
                if current_note is not None and note_start is not None and len(freq_accumulator) > 0:
                    duration = time - note_start
                    # Use median frequency for robustness
                    avg_freq = np.median(freq_accumulator)

                    # Only save notes with minimum duration (configurable)
                    if duration >= self.min_note_duration:
                        notes.append({
                            'time': note_start,
                            'frequency': avg_freq,
                            'note': current_note,
                            'duration': duration
                        })
                    current_note = None
                    note_start = None
                    freq_accumulator = []

        # Don't forget to save the last note if it exists
        if current_note is not None and note_start is not None and len(freq_accumulator) > 0:
            duration = times[-1] - note_start
            avg_freq = np.median(freq_accumulator)
            if duration >= self.min_note_duration:
                notes.append({
                    'time': note_start,
                    'frequency': avg_freq,
                    'note': current_note,
                    'duration': duration
                })

        return notes, times, f0


class AudioSynthPlayer(QThread):
    """Background thread for synthesized audio playback of detected notes"""
    progress = pyqtSignal(str)  # Progress message
    finished = pyqtSignal()
    error = pyqtSignal(str)
    note_playing = pyqtSignal(str)  # Currently playing note for piano visualization

    def __init__(self):
        super().__init__()
        self.notes_to_play = []
        self.tempo_multiplier = 1.0
        self.stop_flag = False
        self.sample_rate = 44100  # CD quality for better sound

    def load_notes(self, notes, tempo_multiplier=1.0):
        """Load notes to play"""
        self.notes_to_play = notes
        self.tempo_multiplier = tempo_multiplier
        self.stop_flag = False

    def generate_note_audio(self, frequency, duration, sample_rate=44100):
        """Generate plucked string sound using Karplus-Strong algorithm

        This produces a guitar/harp-like sound that is much more pleasant than sine waves.
        No new dependencies required - uses only numpy.
        """
        # Number of samples
        num_samples = int(duration * sample_rate)

        if num_samples == 0:
            return np.array([], dtype=np.float32)

        # Calculate delay length (period of the frequency)
        delay_samples = int(sample_rate / frequency)

        # Ensure minimum delay line length
        if delay_samples < 2:
            delay_samples = 2

        # Initialize delay line with noise burst (the "pluck")
        # This simulates plucking a string
        delay_line = np.random.uniform(-0.5, 0.5, delay_samples)

        # Generate samples using Karplus-Strong feedback loop
        output = np.zeros(num_samples)

        for i in range(num_samples):
            # Output current sample from delay line
            output[i] = delay_line[0]

            # Apply lowpass filter for natural decay (averaging)
            # The 0.996 factor controls the decay rate
            new_sample = 0.996 * (delay_line[0] + delay_line[1]) / 2

            # Shift delay line and add feedback
            delay_line = np.roll(delay_line, -1)
            delay_line[-1] = new_sample

        # Apply overall envelope for more natural sound
        # Attack (10ms ramp up to smooth the initial pluck)
        attack_samples = int(0.01 * sample_rate)
        if num_samples > attack_samples and attack_samples > 0:
            attack_envelope = np.linspace(0, 1, attack_samples)
            output[:attack_samples] *= attack_envelope

        # Release (100ms ramp down at end for smooth ending)
        release_samples = int(0.1 * sample_rate)
        if num_samples > release_samples and release_samples > 0:
            release_envelope = np.linspace(1, 0, release_samples)
            output[-release_samples:] *= release_envelope

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0.001:  # Avoid division by very small numbers
            output = output / max_val

        # Reduce volume to comfortable listening level
        output *= 0.6

        return output.astype(np.float32)

    def run(self):
        """Play notes using synthesized audio"""
        try:
            import sounddevice as sd

            self.progress.emit(f"Playing {len(self.notes_to_play)} notes...")

            # Minimum note duration for audibility (in seconds)
            min_note_duration = 0.2  # 200ms minimum so you can hear the pitch

            # Play each note
            for i, note in enumerate(self.notes_to_play):
                if self.stop_flag:
                    break

                # Get frequency from note data
                frequency = note.get('frequency')
                if frequency is None or np.isnan(frequency):
                    continue  # Skip invalid notes

                # Calculate duration with tempo adjustment
                original_duration = note['duration'] / self.tempo_multiplier

                # Use minimum duration for audibility, but keep original for gap calculation
                playback_duration = max(original_duration, min_note_duration)

                # Generate audio for this note with extended duration
                note_audio = self.generate_note_audio(frequency, playback_duration, self.sample_rate)

                # Skip if no audio generated
                if len(note_audio) == 0:
                    continue

                # Update progress and emit note for piano visualization
                self.progress.emit(f"Playing note {i+1}/{len(self.notes_to_play)}: {note['note']} ({frequency:.1f} Hz)")
                self.note_playing.emit(note['note'])  # Signal for piano keyboard

                # Play the note with proper buffer settings
                sd.play(note_audio, self.sample_rate, blocksize=4096, device=None)

                # Wait for playback to complete
                sd.wait()

                # Calculate gap to next note (based on original timing)
                if i < len(self.notes_to_play) - 1 and not self.stop_flag:
                    next_note = self.notes_to_play[i + 1]
                    # Gap is based on original note timing, not extended playback
                    original_gap = (next_note['time'] - (note['time'] + note['duration'])) / self.tempo_multiplier

                    # Use a small fixed gap if notes overlap or are very close
                    gap = max(0.05, original_gap)  # At least 50ms gap
                    time.sleep(gap)

            if not self.stop_flag:
                self.progress.emit("Playback complete!")
                self.finished.emit()
            else:
                self.progress.emit("Playback stopped")
                sd.stop()

        except ImportError:
            self.error.emit("sounddevice not installed. Please run:\npip install sounddevice")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.error.emit(f"Audio playback error: {str(e)}\n\nDetails:\n{error_details}")

    def stop(self):
        """Stop audio playback"""
        self.stop_flag = True
        try:
            import sounddevice as sd
            sd.stop()
        except:
            pass


class AudioPlayer(QThread):
    """Background thread for audio playback"""
    position_changed = pyqtSignal(float)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.audio = None
        self.sr = None
        self.is_playing = False
        self.current_position = 0
        self.stop_flag = False
        self.stream = None

    def load_audio(self, audio, sr):
        self.audio = audio
        self.sr = sr
        self.current_position = 0

    def run(self):
        """Play audio using sounddevice"""
        try:
            import sounddevice as sd

            if self.audio is None:
                return

            # Configure for smooth playback
            # Larger blocksize reduces stuttering but increases latency
            sd.default.blocksize = 8192  # Increased further for even smoother playback
            sd.default.latency = 'high'  # Prioritize smooth playback over low latency
            sd.default.device = None  # Use default device

            # Start from current position
            start_sample = int(self.current_position * self.sr)
            audio_chunk = self.audio[start_sample:]

            # Reset stop flag
            self.stop_flag = False

            # Play audio with blocking=False and capture the stream
            sd.play(audio_chunk, self.sr, blocking=False)
            stream = sd.get_stream()  # Get the actual stream object

            # Track start time for position calculation
            import time
            playback_start_time = time.time()
            initial_position = start_sample / self.sr

            # Update position while playing (less frequently to reduce overhead)
            while not self.stop_flag:
                # Check if still playing using the captured stream
                if not stream.active:
                    break

                # Calculate current position based on elapsed time
                # This is more reliable than querying the stream
                elapsed_time = time.time() - playback_start_time
                self.current_position = initial_position + elapsed_time
                self.position_changed.emit(self.current_position)

                # Update every 200ms for smoother visual feedback without overhead
                self.msleep(200)

            # Wait for playback to finish if not stopped
            if not self.stop_flag:
                sd.wait()
                self.finished.emit()
            else:
                sd.stop()

        except ImportError as e:
            error_msg = "Audio playback not available. Please install sounddevice:\npip install sounddevice"
            print(error_msg)
            self.error_occurred.emit(error_msg)
        except Exception as e:
            error_msg = f"Playback error: {str(e)}\n\nTry installing PortAudio or reinstalling sounddevice."
            print(error_msg)
            self.error_occurred.emit(error_msg)

    def stop(self):
        """Stop audio playback"""
        self.stop_flag = True
        try:
            import sounddevice as sd
            sd.stop()
        except:
            pass


class PianoWidget(FigureCanvas):
    """Widget for displaying piano keyboard with note visualization"""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(12, 2))
        super().__init__(self.figure)
        self.setParent(parent)

        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()

        # Piano configuration
        self.octave_range = (2, 6)  # C2 to C6
        self.active_notes = set()  # Currently playing notes

        self.draw_piano()

    def draw_piano(self):
        """Draw the piano keyboard"""
        self.ax.clear()

        # Note names and colors
        white_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        black_notes = ['C#', 'D#', 'F#', 'G#', 'A#']

        white_key_width = 1
        black_key_width = 0.6
        white_key_height = 4
        black_key_height = 2.5

        x_position = 0
        key_positions = {}  # Store positions for highlighting

        # Draw keys for each octave
        for octave in range(self.octave_range[0], self.octave_range[1] + 1):
            for note in white_notes:
                note_name = f"{note}{octave}"

                # Determine color (active vs inactive)
                color = 'yellow' if note_name in self.active_notes else 'white'
                edge_color = 'red' if note_name in self.active_notes else 'black'
                linewidth = 2 if note_name in self.active_notes else 1

                # Draw white key
                rect = plt.Rectangle((x_position, 0), white_key_width, white_key_height,
                                    facecolor=color, edgecolor=edge_color, linewidth=linewidth)
                self.ax.add_patch(rect)

                # Add note label
                if octave == 4:  # Label middle octave only
                    self.ax.text(x_position + white_key_width/2, 0.3, note,
                               ha='center', va='bottom', fontsize=8)

                key_positions[note_name] = (x_position + white_key_width/2, white_key_height/2)
                x_position += white_key_width

        # Draw black keys on top
        x_position = 0
        for octave in range(self.octave_range[0], self.octave_range[1] + 1):
            for i, note in enumerate(white_notes[:-1]):  # Skip last B
                x_position += white_key_width

                # Check if there's a black key after this white key
                if note in ['C', 'D', 'F', 'G', 'A']:
                    black_note = note + '#'
                    note_name = f"{black_note}{octave}"

                    # Determine color
                    color = 'yellow' if note_name in self.active_notes else 'black'
                    edge_color = 'red' if note_name in self.active_notes else 'black'
                    linewidth = 2 if note_name in self.active_notes else 1

                    # Draw black key (offset to left)
                    rect = plt.Rectangle((x_position - black_key_width/2, white_key_height - black_key_height),
                                        black_key_width, black_key_height,
                                        facecolor=color, edgecolor=edge_color, linewidth=linewidth, zorder=2)
                    self.ax.add_patch(rect)

                    key_positions[note_name] = (x_position, white_key_height - black_key_height/2)

            x_position += white_key_width  # Last B

        self.ax.set_xlim(-0.5, x_position + 0.5)
        self.ax.set_ylim(-0.5, white_key_height + 0.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_title('Piano Keyboard - Notes will light up during playback', fontsize=10)

        self.draw()

    def highlight_notes(self, notes):
        """Highlight specific notes on the keyboard"""
        self.active_notes = set(notes)
        self.draw_piano()


class VisualizationWidget(FigureCanvas):
    """Widget for displaying pitch visualization"""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 6))
        super().__init__(self.figure)
        self.setParent(parent)

        # Create subplots
        self.ax1 = self.figure.add_subplot(211)  # Waveform
        self.ax2 = self.figure.add_subplot(212)  # Pitch

        self.figure.tight_layout(pad=3.0)

        # Initialize data
        self.times = None
        self.audio = None
        self.f0 = None
        self.notes = None
        self.runs = []
        self.current_time = 0

        # Store position line references for fast updates
        self.position_line1 = None
        self.position_line2 = None

        # Cache backgrounds for blit rendering
        self.background1 = None
        self.background2 = None

    def update_data(self, audio, sr, times, f0, notes):
        """Update visualization with new data"""
        self.audio = audio
        self.sr = sr
        self.times = times
        self.f0 = f0
        self.notes = notes
        self.plot()

    def plot(self):
        """Plot waveform and pitch"""
        if self.audio is None:
            return

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()

        # Plot waveform
        time_axis = np.linspace(0, len(self.audio) / self.sr, len(self.audio))
        self.ax1.plot(time_axis, self.audio, linewidth=0.5, alpha=0.7)
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_title('Waveform')
        self.ax1.set_xlim(0, len(self.audio) / self.sr)

        # Plot current position line and store reference (animated for blit)
        self.position_line1 = self.ax1.axvline(x=self.current_time, color='r', linestyle='--', linewidth=2, animated=True)

        # Plot pitch
        if self.f0 is not None and self.times is not None:
            # Remove NaN values for plotting
            mask = ~np.isnan(self.f0)
            self.ax2.plot(self.times[mask], self.f0[mask], 'b-', linewidth=2, label='Detected Pitch')

            # Plot notes as colored regions
            if self.notes:
                colors = plt.cm.rainbow(np.linspace(0, 1, len(self.notes)))
                for i, note in enumerate(self.notes):
                    self.ax2.axvspan(
                        note['time'],
                        note['time'] + note['duration'],
                        alpha=0.3,
                        color=colors[i]
                    )
                    # Add note label
                    self.ax2.text(
                        note['time'] + note['duration']/2,
                        note['frequency'],
                        note['note'],
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        fontweight='bold'
                    )

        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Frequency (Hz)')
        self.ax2.set_title('Detected Pitch and Notes (Melody)')

        # Auto-zoom based on data range with padding
        if self.times is not None and len(self.times) > 0:
            time_min = np.min(self.times)
            time_max = np.max(self.times)
            time_range = time_max - time_min
            padding = time_range * 0.05 if time_range > 0 else 0.5  # 5% padding or 0.5s minimum
            self.ax2.set_xlim(max(0, time_min - padding), time_max + padding)
        else:
            self.ax2.set_xlim(0, len(self.audio) / self.sr if self.audio is not None else 10)

        self.ax2.legend()

        # Plot current position line and store reference (animated for blit)
        self.position_line2 = self.ax2.axvline(x=self.current_time, color='r', linestyle='--', linewidth=2, animated=True)

        # Draw the figure first
        self.draw()

        # Cache the backgrounds for blit rendering (do this AFTER draw)
        # This allows us to restore the background quickly without redrawing everything
        try:
            self.background1 = self.figure.canvas.copy_from_bbox(self.ax1.bbox)
            self.background2 = self.figure.canvas.copy_from_bbox(self.ax2.bbox)
        except:
            # If caching fails, that's ok, we'll fall back to regular draw
            pass

    def update_position(self, time):
        """Update current playback position - optimized with blit rendering"""
        self.current_time = time

        # Only update if we have position lines (after initial plot)
        if self.position_line1 is not None and self.position_line2 is not None:
            # Try ultra-fast blit rendering first
            if self.background1 is not None and self.background2 is not None:
                try:
                    # Update position lines data
                    self.position_line1.set_xdata([time, time])
                    self.position_line2.set_xdata([time, time])

                    # Restore cached backgrounds (fast!)
                    self.figure.canvas.restore_region(self.background1)
                    self.figure.canvas.restore_region(self.background2)

                    # Draw only the animated artists (position lines)
                    self.ax1.draw_artist(self.position_line1)
                    self.ax2.draw_artist(self.position_line2)

                    # Blit only the changed regions (extremely fast!)
                    self.figure.canvas.blit(self.ax1.bbox)
                    self.figure.canvas.blit(self.ax2.bbox)

                    return  # Success! Exit early
                except:
                    # Blit failed, fall through to regular draw
                    pass

            # Fallback: regular update (slower but reliable)
            try:
                self.position_line1.set_xdata([time, time])
                self.position_line2.set_xdata([time, time])
                self.draw()
            except:
                # If even that fails, do nothing (prevents crashes during resize, etc.)
                pass
        else:
            # Initial plot
            self.plot()


class VocalCoachApp(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.audio = None
        self.sr = None
        self.duration = 0
        self.pitch_detector = PitchDetector()
        self.audio_player = AudioPlayer()
        self.note_player = AudioSynthPlayer()
        self.is_playing = False
        self.is_playing_notes = False
        self.notes_data = []
        self.bpm = 0.0  # Detected BPM

        # Connect player signals
        self.audio_player.position_changed.connect(self.on_position_changed)
        self.audio_player.finished.connect(self.on_playback_finished)
        self.audio_player.error_occurred.connect(self.on_playback_error)

        # Connect note player signals
        self.note_player.progress.connect(self.on_note_playback_progress)
        self.note_player.finished.connect(self.on_note_playback_finished)
        self.note_player.error.connect(self.on_note_playback_error)
        self.note_player.note_playing.connect(self.on_note_playing)

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Vocal Riffs and Runs Coach')
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Title
        title = QLabel('Vocal Pitch Detection & Analysis')
        title.setFont(QFont('Arial', 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # File controls
        file_layout = QHBoxLayout()
        self.load_btn = QPushButton('📁 Load Audio File')
        self.load_btn.clicked.connect(self.load_audio_file)
        self.file_label = QLabel('No file loaded')
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.file_label)
        file_layout.addStretch()
        layout.addLayout(file_layout)

        # YouTube controls
        from PyQt5.QtWidgets import QLineEdit
        youtube_layout = QHBoxLayout()
        youtube_label = QLabel('Or YouTube URL:')
        self.youtube_input = QLineEdit()
        self.youtube_input.setPlaceholderText('https://www.youtube.com/watch?v=...')
        self.youtube_btn = QPushButton('🎵 Download from YouTube')
        self.youtube_btn.clicked.connect(self.download_from_youtube)
        youtube_layout.addWidget(youtube_label)
        youtube_layout.addWidget(self.youtube_input)
        youtube_layout.addWidget(self.youtube_btn)
        layout.addLayout(youtube_layout)

        # Download progress label
        self.download_label = QLabel('')
        self.download_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.download_label)

        # Piano keyboard visualization
        self.piano_widget = PianoWidget(self)
        layout.addWidget(self.piano_widget)

        # Visualization
        self.viz_widget = VisualizationWidget(self)
        layout.addWidget(self.viz_widget)

        # Progress bar
        self.progress_bar = QSlider(Qt.Horizontal)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1000)
        self.progress_bar.sliderPressed.connect(self.on_slider_pressed)
        self.progress_bar.sliderReleased.connect(self.on_slider_released)
        layout.addWidget(self.progress_bar)

        # Time and BPM label
        time_bpm_layout = QHBoxLayout()
        self.time_label = QLabel('0:00 / 0:00')
        self.time_label.setAlignment(Qt.AlignCenter)
        time_bpm_layout.addStretch()
        time_bpm_layout.addWidget(self.time_label)
        time_bpm_layout.addStretch()

        self.bpm_label = QLabel('BPM: --')
        self.bpm_label.setAlignment(Qt.AlignCenter)
        self.bpm_label.setFont(QFont('Arial', 10, QFont.Bold))
        time_bpm_layout.addWidget(self.bpm_label)
        time_bpm_layout.addStretch()
        layout.addLayout(time_bpm_layout)

        # Playback controls
        controls_layout = QHBoxLayout()

        self.play_btn = QPushButton('▶ Play')
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)

        self.stop_btn = QPushButton('⏹ Stop')
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)

        self.analyze_btn = QPushButton('🔍 Analyze Pitch')
        self.analyze_btn.clicked.connect(self.analyze_pitch)
        self.analyze_btn.setEnabled(False)

        self.detect_bpm_btn = QPushButton('🎵 Detect BPM')
        self.detect_bpm_btn.clicked.connect(self.detect_bpm)
        self.detect_bpm_btn.setEnabled(False)
        self.detect_bpm_btn.setToolTip('Detect the tempo/BPM of the song')

        self.export_btn = QPushButton('💾 Export Notes')
        self.export_btn.clicked.connect(self.export_notes)
        self.export_btn.setEnabled(False)

        controls_layout.addStretch()
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.analyze_btn)
        controls_layout.addWidget(self.detect_bpm_btn)
        controls_layout.addWidget(self.export_btn)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Analysis progress bar
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setMinimum(0)
        self.analysis_progress.setMaximum(100)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setTextVisible(True)
        self.analysis_progress.setFormat("Ready")
        self.analysis_progress.setVisible(False)  # Hidden until analysis starts
        layout.addWidget(self.analysis_progress)

        # Pitch Detection Settings (new section above run detection)
        pitch_settings_group = QGroupBox('Pitch Detection Settings')
        pitch_settings_layout = QVBoxLayout()

        # First row: Algorithm and basic settings
        pitch_row1 = QHBoxLayout()

        # Algorithm selection
        pitch_row1.addWidget(QLabel('Algorithm:'))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem('pYIN (Probabilistic)', 'pyin')
        self.algorithm_combo.addItem('YIN (Simple)', 'yin')
        self.algorithm_combo.addItem('CREPE (Deep Learning)', 'crepe')
        self.algorithm_combo.setToolTip('Pitch detection algorithm\npYIN: Better for noisy audio, more robust\nYIN: Simpler, sometimes more accurate for clean vocals\nCREPE: Deep learning model, most accurate (90-95%), slower')
        pitch_row1.addWidget(self.algorithm_combo)

        # Voiced confidence threshold
        pitch_row1.addWidget(QLabel('Voice Confidence:'))
        self.voiced_confidence_spin = QDoubleSpinBox()
        self.voiced_confidence_spin.setRange(0.0, 1.0)
        self.voiced_confidence_spin.setValue(0.75)
        self.voiced_confidence_spin.setSingleStep(0.05)
        self.voiced_confidence_spin.setDecimals(2)
        self.voiced_confidence_spin.setToolTip('Minimum confidence to detect a note (0.0-1.0)\nHigher = fewer false positives, may miss quiet notes\nLower = more sensitive, may detect noise')
        pitch_row1.addWidget(self.voiced_confidence_spin)

        # Energy threshold
        pitch_row1.addWidget(QLabel('Energy Threshold:'))
        self.energy_threshold_spin = QDoubleSpinBox()
        self.energy_threshold_spin.setRange(0.0, 0.5)
        self.energy_threshold_spin.setValue(0.15)
        self.energy_threshold_spin.setSingleStep(0.05)
        self.energy_threshold_spin.setDecimals(2)
        self.energy_threshold_spin.setToolTip('Minimum audio energy to detect notes (0.0-0.5)\nHigher = ignores quiet sections\nLower = more sensitive to soft vocals')
        pitch_row1.addWidget(self.energy_threshold_spin)

        # Min note duration
        pitch_row1.addWidget(QLabel('Min Note (ms):'))
        self.min_note_duration_spin = QSpinBox()
        self.min_note_duration_spin.setRange(20, 500)
        self.min_note_duration_spin.setValue(50)
        self.min_note_duration_spin.setSingleStep(10)
        self.min_note_duration_spin.setSuffix('ms')
        self.min_note_duration_spin.setToolTip('Minimum note duration in milliseconds\nHigher = filters out very short detections (noise)\nLower = captures faster runs')
        pitch_row1.addWidget(self.min_note_duration_spin)

        # Vocal isolation toggle
        self.isolate_vocals_checkbox = QCheckBox('Isolate Vocals')
        self.isolate_vocals_checkbox.setChecked(False)  # Default OFF for a cappella
        self.isolate_vocals_checkbox.setToolTip('Apply vocal isolation before pitch detection\nUse for songs with background music\nDisable for a cappella tracks')
        pitch_row1.addWidget(self.isolate_vocals_checkbox)

        pitch_row1.addStretch()
        pitch_settings_layout.addLayout(pitch_row1)

        # Second row: Region selection
        pitch_row2 = QHBoxLayout()

        pitch_row2.addWidget(QLabel('Analyze Region:'))

        pitch_row2.addWidget(QLabel('Start (s):'))
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(0.0, 9999.0)
        self.start_time_spin.setValue(0.0)
        self.start_time_spin.setSingleStep(1.0)
        self.start_time_spin.setDecimals(1)
        self.start_time_spin.setToolTip('Start time for analysis (0 = beginning)')
        pitch_row2.addWidget(self.start_time_spin)

        pitch_row2.addWidget(QLabel('End (s):'))
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(0.0, 9999.0)
        self.end_time_spin.setValue(0.0)
        self.end_time_spin.setSingleStep(1.0)
        self.end_time_spin.setDecimals(1)
        self.end_time_spin.setSpecialValueText('End')
        self.end_time_spin.setToolTip('End time for analysis (0 = full track)')
        pitch_row2.addWidget(self.end_time_spin)

        self.set_start_btn = QPushButton('← Set Start')
        self.set_start_btn.clicked.connect(self.set_start_to_current_position)
        self.set_start_btn.setToolTip('Set start time to current playback position')
        pitch_row2.addWidget(self.set_start_btn)

        self.set_end_btn = QPushButton('Set End →')
        self.set_end_btn.clicked.connect(self.set_end_to_current_position)
        self.set_end_btn.setToolTip('Set end time to current playback position')
        pitch_row2.addWidget(self.set_end_btn)

        self.reset_region_btn = QPushButton('Reset to Full Track')
        self.reset_region_btn.clicked.connect(self.reset_analysis_region)
        pitch_row2.addWidget(self.reset_region_btn)

        pitch_row2.addStretch()
        pitch_settings_layout.addLayout(pitch_row2)

        pitch_settings_group.setLayout(pitch_settings_layout)
        layout.addWidget(pitch_settings_group)



        # Note Playback controls
        playback_group = QGroupBox('Note Playback - Hear Detected Notes')
        playback_layout = QHBoxLayout()

        # Playback speed control
        playback_layout.addWidget(QLabel('Playback Speed:'))
        self.tempo_spin = QDoubleSpinBox()
        self.tempo_spin.setRange(0.25, 4.0)
        self.tempo_spin.setValue(1.0)
        self.tempo_spin.setSingleStep(0.25)
        self.tempo_spin.setDecimals(2)
        self.tempo_spin.setSuffix('x')
        self.tempo_spin.setToolTip('Playback speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)')
        playback_layout.addWidget(self.tempo_spin)

        playback_layout.addSpacing(20)

        # Play all notes button
        self.play_all_notes_btn = QPushButton('🎹 Play All Notes')
        self.play_all_notes_btn.clicked.connect(self.play_all_notes)
        self.play_all_notes_btn.setEnabled(False)
        self.play_all_notes_btn.setToolTip('Play back all detected notes as synthesized audio')
        playback_layout.addWidget(self.play_all_notes_btn)

        # Stop button
        self.stop_notes_btn = QPushButton('⏹ Stop Notes')
        self.stop_notes_btn.clicked.connect(self.stop_note_playback)
        self.stop_notes_btn.setEnabled(False)
        playback_layout.addWidget(self.stop_notes_btn)

        # Status label
        self.note_status_label = QLabel('')
        self.note_status_label.setAlignment(Qt.AlignCenter)
        playback_layout.addWidget(self.note_status_label)

        playback_layout.addStretch()
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)


        # Status bar
        self.statusBar().showMessage('Ready')

    def load_audio_file(self):
        """Load an audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Open Audio File',
            '',
            'Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*.*)'
        )

        if file_path:
            try:
                self.statusBar().showMessage(f'Loading {os.path.basename(file_path)}...')
                QApplication.processEvents()

                # Load audio
                self.audio, self.sr = librosa.load(file_path, sr=None, mono=True)
                self.duration = len(self.audio) / self.sr

                # Update UI
                self.file_label.setText(f'Loaded: {os.path.basename(file_path)}')
                self.play_btn.setEnabled(True)
                self.analyze_btn.setEnabled(True)
                self.detect_bpm_btn.setEnabled(True)
                self.progress_bar.setValue(0)

                # Update time label
                self.update_time_label(0)

                # Load audio into player
                self.audio_player.load_audio(self.audio, self.sr)

                # Display waveform
                self.viz_widget.update_data(self.audio, self.sr, None, None, None)

                self.statusBar().showMessage(f'Loaded {os.path.basename(file_path)} successfully', 3000)

            except Exception as e:
                self.statusBar().showMessage(f'Error loading file: {str(e)}')
                print(f"Error loading audio: {e}")

    def download_from_youtube(self):
        """Download audio from YouTube URL"""
        url = self.youtube_input.text().strip()

        if not url:
            self.statusBar().showMessage('Please enter a YouTube URL')
            return

        # Basic URL validation
        if 'youtube.com' not in url and 'youtu.be' not in url:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Invalid URL', 'Please enter a valid YouTube URL')
            return

        # Disable button during download
        self.youtube_btn.setEnabled(False)
        self.download_label.setText('Preparing download...')

        # Create and start downloader
        self.youtube_downloader = YouTubeDownloader(url)
        self.youtube_downloader.progress.connect(self.on_download_progress)
        self.youtube_downloader.finished.connect(self.on_download_finished)
        self.youtube_downloader.error.connect(self.on_download_error)
        self.youtube_downloader.start()

    def on_download_progress(self, message):
        """Handle download progress updates"""
        self.download_label.setText(message)
        self.statusBar().showMessage(message)

    def on_download_finished(self, file_path):
        """Handle successful download"""
        self.youtube_btn.setEnabled(True)
        self.download_label.setText('✅ Download complete!')

        # Load the downloaded audio
        try:
            self.statusBar().showMessage(f'Loading downloaded audio...')
            QApplication.processEvents()

            # Load audio
            self.audio, self.sr = librosa.load(file_path, sr=None, mono=True)
            self.duration = len(self.audio) / self.sr

            # Update UI
            filename = os.path.basename(file_path)
            self.file_label.setText(f'Loaded from YouTube: {filename}')
            self.play_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)
            self.detect_bpm_btn.setEnabled(True)
            self.progress_bar.setValue(0)

            # Update time label
            self.update_time_label(0)

            # Load audio into player
            self.audio_player.load_audio(self.audio, self.sr)

            # Display waveform
            self.viz_widget.update_data(self.audio, self.sr, None, None, None)

            self.statusBar().showMessage(f'Ready to analyze!', 3000)

            # Clear download label after a few seconds
            QTimer.singleShot(3000, lambda: self.download_label.setText(''))

        except Exception as e:
            self.statusBar().showMessage(f'Error loading downloaded file: {str(e)}')
            self.download_label.setText(f'❌ Error loading file')
            print(f"Error loading audio: {e}")

    def on_download_error(self, error_msg):
        """Handle download errors"""
        from PyQt5.QtWidgets import QMessageBox

        self.youtube_btn.setEnabled(True)
        self.download_label.setText('❌ Download failed')

        # Show error dialog
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle('Download Error')
        msg_box.setText('Failed to download from YouTube')
        msg_box.setInformativeText(error_msg)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

        self.statusBar().showMessage('Download failed')

        # Clear download label after a few seconds
        QTimer.singleShot(3000, lambda: self.download_label.setText(''))

    def reset_analysis_region(self):
        """Reset the analysis region to full track"""
        self.start_time_spin.setValue(0.0)
        self.end_time_spin.setValue(0.0)

    def set_start_to_current_position(self):
        """Set the start time to current playback position"""
        if hasattr(self, 'audio_player'):
            current_pos = self.audio_player.current_position
            self.start_time_spin.setValue(round(current_pos, 1))
            self.statusBar().showMessage(f'Start time set to {current_pos:.1f}s', 2000)

    def set_end_to_current_position(self):
        """Set the end time to current playback position"""
        if hasattr(self, 'audio_player'):
            current_pos = self.audio_player.current_position
            self.end_time_spin.setValue(round(current_pos, 1))
            self.statusBar().showMessage(f'End time set to {current_pos:.1f}s', 2000)

    def analyze_pitch(self):
        """Analyze pitch from the loaded audio"""
        if self.audio is None:
            return

        # Disable analyze button during analysis
        self.analyze_btn.setEnabled(False)
        self.statusBar().showMessage('Starting pitch analysis...')

        # Show and reset progress bar
        self.analysis_progress.setVisible(True)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setFormat("Starting analysis...")

        # Get pitch detection parameters
        voiced_threshold = self.voiced_confidence_spin.value()
        energy_threshold = self.energy_threshold_spin.value()
        isolate_vocals = self.isolate_vocals_checkbox.isChecked()
        algorithm = self.algorithm_combo.currentData()
        min_note_duration = self.min_note_duration_spin.value() / 1000.0  # Convert ms to seconds

        # Get region selection
        start_time = self.start_time_spin.value() if self.start_time_spin.value() > 0 else None
        end_time = self.end_time_spin.value() if self.end_time_spin.value() > 0 else None

        # Validate region
        if end_time is not None and start_time is not None and end_time <= start_time:
            self.statusBar().showMessage('Error: End time must be greater than start time')
            self.analyze_btn.setEnabled(True)
            self.analysis_progress.setVisible(False)
            return

        # Create and start analysis thread
        self.analysis_thread = PitchAnalysisThread(
            self.audio, self.sr,
            voiced_threshold, energy_threshold, isolate_vocals,
            start_time, end_time, algorithm, min_note_duration
        )
        self.analysis_thread.progress.connect(self.on_analysis_progress)
        self.analysis_thread.progress_percent.connect(self.on_analysis_progress_percent)
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.start()

    def on_analysis_progress(self, message):
        """Handle analysis progress updates"""
        self.statusBar().showMessage(message)
        self.analysis_progress.setFormat(message)

    def on_analysis_progress_percent(self, percent):
        """Handle analysis progress percentage updates"""
        self.analysis_progress.setValue(percent)

    def on_analysis_finished(self, notes, times, f0):
        """Handle completed analysis"""
        self.analyze_btn.setEnabled(True)

        # Hide progress bar after brief delay
        QTimer.singleShot(2000, lambda: self.analysis_progress.setVisible(False))

        # Store notes data
        self.notes_data = notes

        # Store times and f0 for reanalysis
        self.times = times
        self.f0 = f0

        # Update visualization
        self.viz_widget.update_data(self.audio, self.sr, times, f0, notes)

        # Enable export
        self.export_btn.setEnabled(True)

        # Enable note playback for all notes (always available after analysis)
        if notes:
            self.play_all_notes_btn.setEnabled(True)

        status_msg = f'Analysis complete! Detected {len(notes)} notes in the melody'
        self.statusBar().showMessage(status_msg, 3000)

    def on_analysis_error(self, error_msg):
        """Handle analysis errors"""
        self.analyze_btn.setEnabled(True)
        self.analysis_progress.setVisible(False)
        self.statusBar().showMessage(f'Error analyzing pitch: {error_msg}')
        print(f"Error analyzing pitch: {error_msg}")

    def detect_bpm(self):
        """Detect BPM from the loaded audio"""
        if self.audio is None:
            return

        # Disable BPM button during detection
        self.detect_bpm_btn.setEnabled(False)
        self.statusBar().showMessage('Detecting BPM...')

        # Create and start BPM detection thread
        self.bpm_thread = BPMDetectionThread(self.audio, self.sr)
        self.bpm_thread.progress.connect(self.on_bpm_progress)
        self.bpm_thread.finished.connect(self.on_bpm_finished)
        self.bpm_thread.error.connect(self.on_bpm_error)
        self.bpm_thread.start()

    def on_bpm_progress(self, message):
        """Handle BPM detection progress updates"""
        self.statusBar().showMessage(message)

    def on_bpm_finished(self, bpm):
        """Handle completed BPM detection"""
        self.detect_bpm_btn.setEnabled(True)
        self.bpm = bpm
        self.bpm_label.setText(f'BPM: {bpm:.1f}')
        self.statusBar().showMessage(f'BPM detected: {bpm:.1f}', 3000)

    def on_bpm_error(self, error_msg):
        """Handle BPM detection errors"""
        self.detect_bpm_btn.setEnabled(True)
        self.bpm_label.setText('BPM: --')
        self.statusBar().showMessage(f'Error detecting BPM: {error_msg}')
        print(f"Error detecting BPM: {error_msg}")

    def toggle_play(self):
        """Toggle play/pause"""
        if not self.is_playing:
            self.play_audio()
        else:
            self.pause_audio()

    def play_audio(self):
        """Start audio playback"""
        if self.audio is None:
            return

        self.is_playing = True
        self.play_btn.setText('⏸ Pause')
        self.stop_btn.setEnabled(True)

        # Start playback thread
        if not self.audio_player.isRunning():
            self.audio_player.stop_flag = False
            self.audio_player.start()

    def pause_audio(self):
        """Pause audio playback"""
        self.is_playing = False
        self.play_btn.setText('▶ Play')
        self.audio_player.stop()

    def stop_playback(self):
        """Stop audio playback"""
        self.is_playing = False
        self.play_btn.setText('▶ Play')
        self.stop_btn.setEnabled(False)

        # Stop player
        self.audio_player.stop()
        self.audio_player.current_position = 0

        # Reset UI
        self.progress_bar.setValue(0)
        self.update_time_label(0)
        self.viz_widget.update_position(0)

    def on_position_changed(self, position):
        """Handle position change during playback"""
        if self.duration > 0:
            progress = int((position / self.duration) * 1000)
            self.progress_bar.setValue(progress)
            self.update_time_label(position)
            self.viz_widget.update_position(position)

    def on_playback_finished(self):
        """Handle playback finished"""
        self.is_playing = False
        self.play_btn.setText('▶ Play')
        self.stop_btn.setEnabled(False)
        self.audio_player.current_position = 0

    def on_playback_error(self, error_msg):
        """Handle playback errors"""
        from PyQt5.QtWidgets import QMessageBox

        self.is_playing = False
        self.play_btn.setText('▶ Play')
        self.stop_btn.setEnabled(False)

        # Show error dialog
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle('Audio Playback Error')
        msg_box.setText('Unable to play audio')
        msg_box.setInformativeText(error_msg)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

        self.statusBar().showMessage('Audio playback error - see details in dialog')

    def on_slider_pressed(self):
        """Handle slider press"""
        if self.is_playing:
            self.pause_audio()

    def on_slider_released(self):
        """Handle slider release (seek)"""
        if self.audio is None:
            return

        # Calculate new position
        progress = self.progress_bar.value() / 1000
        new_position = progress * self.duration

        # Update player position
        self.audio_player.current_position = new_position

        # Update visualization
        self.viz_widget.update_position(new_position)
        self.update_time_label(new_position)

    def update_time_label(self, current_time):
        """Update time label"""
        current_str = self.format_time(current_time)
        duration_str = self.format_time(self.duration)
        self.time_label.setText(f'{current_str} / {duration_str}')

    def format_time(self, seconds):
        """Format seconds as MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f'{minutes}:{secs:02d}'

    def export_notes(self):
        """Export detected notes to a file"""
        if not self.notes_data:
            self.statusBar().showMessage('No notes to export. Please analyze first.')
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Export Notes',
            'detected_notes.txt',
            'Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)'
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if file_path.endswith('.csv'):
                        # CSV format
                        f.write('Time (s),Note,Frequency (Hz),Duration (s)\n')
                        for note in self.notes_data:
                            f.write(f"{note['time']:.3f},{note['note']},{note['frequency']:.2f},{note['duration']:.3f}\n")
                    else:
                        # Human-readable format
                        f.write('Detected Notes\n')
                        f.write('=' * 60 + '\n\n')
                        for i, note in enumerate(self.notes_data, 1):
                            f.write(f"Note {i}:\n")
                            f.write(f"  Time: {note['time']:.3f} s\n")
                            f.write(f"  Note: {note['note']}\n")
                            f.write(f"  Frequency: {note['frequency']:.2f} Hz\n")
                            f.write(f"  Duration: {note['duration']:.3f} s\n")
                            f.write('\n')

                self.statusBar().showMessage(f'Notes exported to {os.path.basename(file_path)}', 3000)

            except Exception as e:
                self.statusBar().showMessage(f'Error exporting notes: {str(e)}')
                print(f"Error exporting notes: {e}")

    def play_all_notes(self):
        """Play all detected notes as synthesized audio"""
        if not self.notes_data:
            self.statusBar().showMessage('No notes to play. Please analyze first.')
            return

        if self.is_playing_notes:
            self.statusBar().showMessage('Note playback already in progress')
            return

        # Get tempo from UI
        tempo = self.tempo_spin.value()

        # Load all notes into note player
        self.note_player.load_notes(self.notes_data, tempo)

        # Update UI
        self.is_playing_notes = True
        self.play_all_notes_btn.setEnabled(False)
        self.stop_notes_btn.setEnabled(True)
        self.note_status_label.setText(f'Playing {len(self.notes_data)} notes...')

        # Start playback
        self.note_player.start()

    def stop_note_playback(self):
        """Stop note playback"""
        self.note_player.stop()
        self.is_playing_notes = False
        self.play_all_notes_btn.setEnabled(True)
        self.stop_notes_btn.setEnabled(False)
        self.note_status_label.setText('Stopped')

    def on_note_playback_progress(self, message):
        """Handle note playback progress updates"""
        self.note_status_label.setText(message)

    def on_note_playing(self, note_name):
        """Handle note currently being played - update piano visualization"""
        self.piano_widget.highlight_notes([note_name])

    def on_note_playback_finished(self):
        """Handle note playback finished"""
        self.is_playing_notes = False
        self.play_all_notes_btn.setEnabled(True)
        self.stop_notes_btn.setEnabled(False)
        self.note_status_label.setText('Playback complete')

        # Clear piano keyboard
        self.piano_widget.highlight_notes([])

        # Clear status after a few seconds
        QTimer.singleShot(3000, lambda: self.note_status_label.setText(''))

    def on_note_playback_error(self, error_msg):
        """Handle note playback errors"""
        self.is_playing_notes = False
        self.play_all_notes_btn.setEnabled(True)
        self.stop_notes_btn.setEnabled(False)

        # Show error dialog
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle('Note Playback Error')
        msg_box.setText('Unable to play notes')
        msg_box.setInformativeText(error_msg)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

        self.note_status_label.setText('Error')
        self.statusBar().showMessage('Note playback error - see dialog for details')


def main():
    app = QApplication(sys.argv)
    window = VocalCoachApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
