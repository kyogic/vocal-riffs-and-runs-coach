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


class PitchAnalysisThread(QThread):
    """Background thread for pitch analysis"""
    progress = pyqtSignal(str)  # Progress message
    finished = pyqtSignal(list, object, object)  # notes, times, f0
    error = pyqtSignal(str)  # Error message

    def __init__(self, audio, sr):
        super().__init__()
        self.audio = audio
        self.sr = sr

    def run(self):
        """Analyze pitch in background"""
        try:
            self.progress.emit('Analyzing pitch...')

            pitch_detector = PitchDetector(self.sr)
            notes, times, f0 = pitch_detector.get_note_data(self.audio)

            self.progress.emit(f'Analysis complete! Detected {len(notes)} notes')
            self.finished.emit(notes, times, f0)

        except Exception as e:
            self.error.emit(f'Analysis error: {str(e)}')


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
            sd.default.blocksize = 4096  # Increased from default for smoother playback
            sd.default.latency = 'high'  # Prioritize smooth playback over low latency

            # Start from current position
            start_sample = int(self.current_position * self.sr)
            audio_chunk = self.audio[start_sample:]

            # Reset stop flag
            self.stop_flag = False

            # Play audio with blocking=False
            sd.play(audio_chunk, self.sr, blocking=False)

            # Track start time for position calculation
            import time
            playback_start_time = time.time()
            initial_position = start_sample / self.sr

            # Update position while playing (less frequently to reduce overhead)
            while not self.stop_flag:
                # Check if still playing
                if not sd.get_stream().active:
                    break

                # Calculate current position based on elapsed time
                # This is more reliable than querying the stream
                elapsed_time = time.time() - playback_start_time
                self.current_position = initial_position + elapsed_time
                self.position_changed.emit(self.current_position)

                # Update every 500ms to minimize GUI impact
                self.msleep(500)

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

        # Store position line references for fast updates
        self.position_line1 = None
        self.position_line2 = None

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

        # Plot current position line and store reference
        self.position_line1 = self.ax1.axvline(x=self.current_time, color='r', linestyle='--', linewidth=2)

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

        # Plot current position line and store reference
        self.position_line2 = self.ax2.axvline(x=self.current_time, color='r', linestyle='--', linewidth=2)

        self.draw()

    def update_position(self, time):
        """Update current playback position - optimized to only move the line"""
        self.current_time = time

        # Only update if we have position lines (after initial plot)
        if self.position_line1 is not None and self.position_line2 is not None:
            try:
                # Update position lines without full redraw
                self.position_line1.set_xdata([time, time])
                self.position_line2.set_xdata([time, time])

                # Draw only the updated parts (much faster)
                self.draw()
            except:
                # If update fails, do full redraw
                self.plot()
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
        self.is_playing = False
        self.notes_data = []

        # Connect player signals
        self.audio_player.position_changed.connect(self.on_position_changed)
        self.audio_player.finished.connect(self.on_playback_finished)
        self.audio_player.error_occurred.connect(self.on_playback_error)

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

    def analyze_pitch(self):
        """Analyze pitch from the loaded audio"""
        if self.audio is None:
            return

        # Disable analyze button during analysis
        self.analyze_btn.setEnabled(False)
        self.statusBar().showMessage('Starting pitch analysis...')

        # Create and start analysis thread
        self.analysis_thread = PitchAnalysisThread(self.audio, self.sr)
        self.analysis_thread.progress.connect(self.on_analysis_progress)
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.start()

    def on_analysis_progress(self, message):
        """Handle analysis progress updates"""
        self.statusBar().showMessage(message)

    def on_analysis_finished(self, notes, times, f0):
        """Handle completed analysis"""
        self.analyze_btn.setEnabled(True)

        # Store notes data
        self.notes_data = notes

        # Update visualization
        self.viz_widget.update_data(self.audio, self.sr, times, f0, notes)

        # Enable export
        self.export_btn.setEnabled(True)

        self.statusBar().showMessage(f'Analysis complete! Detected {len(notes)} notes', 3000)

    def on_analysis_error(self, error_msg):
        """Handle analysis errors"""
        self.analyze_btn.setEnabled(True)
        self.statusBar().showMessage(f'Error analyzing pitch: {error_msg}')
        print(f"Error analyzing pitch: {error_msg}")

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
