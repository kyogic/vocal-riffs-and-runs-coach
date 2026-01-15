#!/usr/bin/env python3
"""
CREPE Pitch Analysis Script (No GUI)
Analyzes audio files using CREPE deep learning model without PyQt5 conflicts
"""

import sys
import os
import argparse
import json
import numpy as np
import librosa
from scipy.ndimage import median_filter


def frequency_to_note(frequency):
    """Convert frequency (Hz) to note name with octave"""
    if frequency <= 0 or np.isnan(frequency):
        return None

    # A4 = 440 Hz is our reference
    A4 = 440.0
    C0 = A4 * pow(2, -4.75)  # C0 is 4.75 octaves below A4

    # Calculate semitone number from C0
    if frequency > 0:
        h = 12 * np.log2(frequency / C0)
        octave = int(h / 12)
        semitone = int(round(h % 12))

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note = note_names[semitone]

        return f"{note}{octave}"
    return None


def analyze_audio_with_crepe(audio_file, min_note_duration=0.03, fmin=80, fmax=800):
    """
    Analyze audio file using CREPE

    Args:
        audio_file: Path to audio file
        min_note_duration: Minimum note duration in seconds (default 30ms)
        fmin: Minimum frequency in Hz (default 80 Hz)
        fmax: Maximum frequency in Hz (default 800 Hz)

    Returns:
        dict with analysis results
    """
    print(f"\nAnalyzing: {audio_file}")
    print("=" * 70)

    # Load audio
    print("Loading audio...")
    audio, sr = librosa.load(audio_file, sr=None, mono=True)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sr} Hz")
    print()

    # Import CREPE (will work here since no PyQt5)
    print("Loading CREPE model...")
    try:
        import crepe
        print(f"  CREPE version: {crepe.__version__ if hasattr(crepe, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"ERROR: CREPE not installed: {e}")
        print("Install with: pip install crepe tensorflow")
        sys.exit(1)
    print()

    # Run CREPE
    print("Running CREPE pitch detection...")
    print("  This may take a minute for longer files...")

    # CREPE expects 16kHz sample rate, resample if needed
    target_sr = 16000
    if sr != target_sr:
        print(f"  Resampling from {sr} Hz to {target_sr} Hz...")
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    else:
        audio_resampled = audio

    # Run CREPE (model_capacity: 'tiny', 'small', 'medium', 'large', 'full')
    time, frequency, confidence, activation = crepe.predict(
        audio_resampled,
        target_sr,
        viterbi=True,  # Use Viterbi decoding for smoother pitch contours
        model_capacity='small',  # Balance between speed and accuracy
        step_size=10  # milliseconds between predictions
    )
    print("  ✓ CREPE analysis complete")
    print()

    # Filter out frequencies outside vocal range
    print(f"Filtering to vocal range: {fmin}-{fmax} Hz...")
    frequency[(frequency < fmin) | (frequency > fmax)] = 0

    # Set unvoiced regions based on confidence threshold
    frequency[confidence < 0.3] = 0

    # Convert to notes
    print("Converting frequencies to notes...")
    notes_data = []
    current_note = None
    note_start = None
    note_frequencies = []

    for i, (t, freq, conf) in enumerate(zip(time, frequency, confidence)):
        if freq > 0 and conf >= 0.3:
            note_name = frequency_to_note(freq)

            if note_name != current_note:
                # Save previous note if long enough
                if current_note is not None and note_frequencies:
                    note_duration = t - note_start
                    if note_duration >= min_note_duration:
                        avg_freq = np.mean(note_frequencies)
                        avg_conf = np.mean([c for _, _, c in
                                          zip(time[max(0, i-len(note_frequencies)):i],
                                              frequency[max(0, i-len(note_frequencies)):i],
                                              confidence[max(0, i-len(note_frequencies)):i])])

                        notes_data.append({
                            'note': current_note,
                            'start': note_start,
                            'end': t,
                            'duration': note_duration,
                            'frequency': avg_freq,
                            'confidence': avg_conf
                        })

                # Start new note
                current_note = note_name
                note_start = t
                note_frequencies = [freq]
            else:
                note_frequencies.append(freq)
        else:
            # Silence - save current note if exists
            if current_note is not None and note_frequencies:
                note_duration = t - note_start
                if note_duration >= min_note_duration:
                    avg_freq = np.mean(note_frequencies)
                    avg_conf = np.mean([c for _, _, c in
                                      zip(time[max(0, i-len(note_frequencies)):i],
                                          frequency[max(0, i-len(note_frequencies)):i],
                                          confidence[max(0, i-len(note_frequencies)):i])])

                    notes_data.append({
                        'note': current_note,
                        'start': note_start,
                        'end': t,
                        'duration': note_duration,
                        'frequency': avg_freq,
                        'confidence': avg_conf
                    })

            current_note = None
            note_start = None
            note_frequencies = []

    # Save last note
    if current_note is not None and note_frequencies:
        note_duration = time[-1] - note_start
        if note_duration >= min_note_duration:
            avg_freq = np.mean(note_frequencies)
            notes_data.append({
                'note': current_note,
                'start': note_start,
                'end': time[-1],
                'duration': note_duration,
                'frequency': avg_freq,
                'confidence': 0.9
            })

    print(f"  ✓ Detected {len(notes_data)} notes")
    print()

    # Detect BPM
    print("Detecting BPM...")
    try:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        bpm = float(tempo) if isinstance(tempo, np.ndarray) else tempo
        print(f"  ✓ BPM: {bpm:.1f}")
    except:
        bpm = None
        print("  ✗ BPM detection failed")
    print()

    return {
        'file': audio_file,
        'duration': duration,
        'sample_rate': sr,
        'bpm': bpm,
        'notes': notes_data,
        'algorithm': 'CREPE',
        'min_note_duration': min_note_duration,
        'fmin': fmin,
        'fmax': fmax
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze audio files using CREPE pitch detection (no GUI conflicts)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.mp3
  %(prog)s song.wav --min-duration 50
  %(prog)s song.mp3 --fmin 100 --fmax 600 --output results.json
        """
    )

    parser.add_argument('audio_file', help='Audio file to analyze')
    parser.add_argument('--min-duration', type=int, default=30,
                       help='Minimum note duration in milliseconds (default: 30)')
    parser.add_argument('--fmin', type=int, default=80,
                       help='Minimum frequency in Hz (default: 80)')
    parser.add_argument('--fmax', type=int, default=800,
                       help='Maximum frequency in Hz (default: 800)')
    parser.add_argument('--output', '-o',
                       help='Output JSON file (default: <audio_file>_crepe.json)')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.audio_file):
        print(f"ERROR: File not found: {args.audio_file}")
        sys.exit(1)

    # Analyze
    min_duration_sec = args.min_duration / 1000.0
    results = analyze_audio_with_crepe(
        args.audio_file,
        min_note_duration=min_duration_sec,
        fmin=args.fmin,
        fmax=args.fmax
    )

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        base = os.path.splitext(args.audio_file)[0]
        output_file = f"{base}_crepe.json"

    # Save results
    print(f"Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total notes detected: {len(results['notes'])}")
    if results['bpm']:
        print(f"BPM: {results['bpm']:.1f}")
    print(f"Results saved to: {output_file}")
    print()

    # Show first few notes
    if results['notes']:
        print("First 10 notes:")
        print("-" * 70)
        print(f"{'Note':<8} {'Start':<10} {'Duration':<12} {'Frequency':<12} {'Confidence'}")
        print("-" * 70)
        for note in results['notes'][:10]:
            print(f"{note['note']:<8} {note['start']:<10.2f} {note['duration']*1000:<12.0f}ms "
                  f"{note['frequency']:<12.1f}Hz {note['confidence']:.2f}")
        if len(results['notes']) > 10:
            print(f"... and {len(results['notes']) - 10} more notes")
        print()


if __name__ == '__main__':
    main()
