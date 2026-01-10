#!/usr/bin/env python3
"""
Vocal Riffs and Runs Coach
A desktop application for vocal pitch detection and analysis
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider,
                             QFileDialog, QProgressBar, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class PitchDetector:
    """Handles pitch detection from audio data"""

    def __init__(self, sr=22050):
        self.sr = sr
        self.hop_length = 512
        self.fmin = librosa.note_to_hz('C2')  # Lowest vocal note
        self.fmax = librosa.note_to_hz('C7')  # Highest vocal note

    def detect_pitch(self, audio):
        """
        Detect pitch from audio using librosa's pyin algorithm
        Returns frequencies, voiced flag, and voiced probabilities
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length
        )
        return f0, voiced_flag, voiced_probs

    def frequency_to_note(self, frequency):
        """Convert frequency to note name"""
        if frequency is None or np.isnan(frequency):
            return None

        note_number = librosa.hz_to_midi(frequency)
        note_name = librosa.midi_to_note(int(round(note_number)))
        return note_name

    def get_note_data(self, audio):
        """
        Analyze audio and return note data with timing
        Returns list of (time, frequency, note_name, duration)
        """
        f0, voiced_flag, voiced_probs = self.detect_pitch(audio)

        # Convert frame indices to time
        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=self.sr,
            hop_length=self.hop_length
        )

        notes = []
        current_note = None
        note_start = None

        for i, (time, freq, voiced) in enumerate(zip(times, f0, voiced_flag)):
            if voiced and not np.isnan(freq):
                note_name = self.frequency_to_note(freq)

                if note_name != current_note:
                    # Save previous note if exists
                    if current_note is not None and note_start is not None:
                        duration = time - note_start
                        notes.append({
                            'time': note_start,
                            'frequency': prev_freq,
                            'note': current_note,
                            'duration': duration
                        })

                    # Start new note
                    current_note = note_name
                    note_start = time
                    prev_freq = freq
            else:
                # End of voiced segment
                if current_note is not None and note_start is not None:
                    duration = time - note_start
                    notes.append({
                        'time': note_start,
                        'frequency': prev_freq,
                        'note': current_note,
                        'duration': duration
                    })
                    current_note = None
                    note_start = None

        return notes, times, f0


class AudioPlayer(QThread):
    """Background thread for audio playback"""
    position_changed = pyqtSignal(float)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.audio = None
        self.sr = None
        self.is_playing = False
        self.current_position = 0
        self.stop_flag = False

    def load_audio(self, audio, sr):
        self.audio = audio
        self.sr = sr
        self.current_position = 0

    def run(self):
        """Play audio using soundfile"""
        try:
            import sounddevice as sd

            if self.audio is None:
                return

            # Start from current position
            start_sample = int(self.current_position * self.sr)

            # Play audio
            sd.play(self.audio[start_sample:], self.sr)

            # Update position while playing
            while sd.get_stream().active and not self.stop_flag:
                current_sample = start_sample + sd.get_stream().time * self.sr
                self.current_position = current_sample / self.sr
                self.position_changed.emit(self.current_position)
                self.msleep(50)

            if not self.stop_flag:
                self.finished.emit()
        except ImportError:
            print("sounddevice not available, playback disabled")
        except Exception as e:
            print(f"Playback error: {e}")

    def stop(self):
        self.stop_flag = True
        try:
            import sounddevice as sd
            sd.stop()
        except:
            pass


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
        self.current_time = 0

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

        # Plot current position line
        self.ax1.axvline(x=self.current_time, color='r', linestyle='--', linewidth=2)

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
        self.ax2.set_title('Detected Pitch and Notes')
        self.ax2.set_xlim(0, len(self.audio) / self.sr if self.audio is not None else 10)
        self.ax2.legend()

        # Plot current position line
        self.ax2.axvline(x=self.current_time, color='r', linestyle='--', linewidth=2)

        self.draw()

    def update_position(self, time):
        """Update current playback position"""
        self.current_time = time
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
        self.is_playing = False
        self.notes_data = []

        # Connect player signals
        self.audio_player.position_changed.connect(self.on_position_changed)
        self.audio_player.finished.connect(self.on_playback_finished)

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
        self.load_btn = QPushButton('Load Audio File')
        self.load_btn.clicked.connect(self.load_audio_file)
        self.file_label = QLabel('No file loaded')
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.file_label)
        file_layout.addStretch()
        layout.addLayout(file_layout)

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

        # Time label
        self.time_label = QLabel('0:00 / 0:00')
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.time_label)

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

        self.export_btn = QPushButton('💾 Export Notes')
        self.export_btn.clicked.connect(self.export_notes)
        self.export_btn.setEnabled(False)

        controls_layout.addStretch()
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.analyze_btn)
        controls_layout.addWidget(self.export_btn)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

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

    def analyze_pitch(self):
        """Analyze pitch from the loaded audio"""
        if self.audio is None:
            return

        try:
            self.statusBar().showMessage('Analyzing pitch...')
            QApplication.processEvents()

            # Detect pitch and notes
            notes, times, f0 = self.pitch_detector.get_note_data(self.audio)
            self.notes_data = notes

            # Update visualization
            self.viz_widget.update_data(self.audio, self.sr, times, f0, notes)

            # Enable export
            self.export_btn.setEnabled(True)

            self.statusBar().showMessage(f'Analysis complete! Detected {len(notes)} notes', 3000)

        except Exception as e:
            self.statusBar().showMessage(f'Error analyzing pitch: {str(e)}')
            print(f"Error analyzing pitch: {e}")

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
                with open(file_path, 'w') as f:
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


def main():
    app = QApplication(sys.argv)
    window = VocalCoachApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
