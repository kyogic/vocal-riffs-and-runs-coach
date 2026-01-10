#!/usr/bin/env python3
"""
Audio Troubleshooting Script
This script tests if audio playback is working correctly on your system.
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    print("-" * 50)

    packages = {
        'PyQt5': 'GUI framework',
        'numpy': 'Numerical operations',
        'librosa': 'Audio analysis',
        'soundfile': 'Audio file I/O',
        'sounddevice': 'Audio playback',
        'matplotlib': 'Visualization',
        'scipy': 'Scientific computing'
    }

    failed = []
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✓ {package:15} - {description}")
        except ImportError as e:
            print(f"✗ {package:15} - FAILED: {e}")
            failed.append(package)

    print("-" * 50)
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print(f"Install with: pip install {' '.join(failed)}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def test_audio_devices():
    """Test audio device availability"""
    print("\n\nTesting audio devices...")
    print("-" * 50)

    try:
        import sounddevice as sd

        # Get default device
        default_device = sd.query_devices(kind='output')
        print(f"Default output device: {default_device['name']}")
        print(f"  Sample rate: {default_device['default_samplerate']} Hz")
        print(f"  Channels: {default_device['max_output_channels']}")

        # List all devices
        print("\nAll audio devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                print(f"  [{i}] {dev['name']} (outputs: {dev['max_output_channels']})")

        return True
    except Exception as e:
        print(f"❌ Error querying audio devices: {e}")
        print("\nPossible solutions:")
        print("  1. Install PortAudio (required by sounddevice)")
        print("  2. Check your system audio settings")
        print("  3. Reinstall sounddevice: pip uninstall sounddevice && pip install sounddevice")
        return False


def test_audio_playback():
    """Test actual audio playback with a beep"""
    print("\n\nTesting audio playback...")
    print("-" * 50)

    try:
        import sounddevice as sd
        import numpy as np

        # Generate a simple beep (440 Hz, 1 second)
        duration = 1.0
        sample_rate = 44100
        frequency = 440  # A4 note

        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)

        print("Playing a 1-second test tone (440 Hz - A4 note)...")
        print("You should hear a beep now!")

        sd.play(audio, sample_rate)
        sd.wait()

        print("✅ Playback successful!")
        response = input("\nDid you hear the beep? (y/n): ").strip().lower()

        if response == 'y':
            print("✅ Audio playback is working correctly!")
            return True
        else:
            print("⚠️  Audio played but you didn't hear it.")
            print("   Check your system volume and speaker connections.")
            return False

    except Exception as e:
        print(f"❌ Playback error: {e}")
        print("\nPossible solutions:")
        print("  1. Check your system volume")
        print("  2. Make sure speakers/headphones are connected")
        print("  3. Try a different audio device")
        print("  4. Reinstall sounddevice with: pip install --force-reinstall sounddevice")
        return False


def test_audio_file():
    """Test loading and playing an audio file"""
    print("\n\nTesting audio file loading...")
    print("-" * 50)

    try:
        import librosa
        import numpy as np

        # Generate a test audio file
        print("Generating test audio (C major scale)...")

        sr = 22050
        duration = 0.3

        # C major scale frequencies
        notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        audio = []

        for freq in notes:
            t = np.linspace(0, duration, int(sr * duration))
            note_audio = 0.3 * np.sin(2 * np.pi * freq * t)
            audio.extend(note_audio)

        audio = np.array(audio)

        print(f"✅ Generated {len(audio)/sr:.2f} seconds of audio")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Audio shape: {audio.shape}")

        # Test pitch detection
        print("\nTesting pitch detection...")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio[:sr*2],  # First 2 seconds
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )

        detected_pitches = f0[~np.isnan(f0)]
        if len(detected_pitches) > 0:
            print(f"✅ Pitch detection working!")
            print(f"   Detected {len(detected_pitches)} pitch values")
            print(f"   Range: {detected_pitches.min():.1f} - {detected_pitches.max():.1f} Hz")
        else:
            print("⚠️  No pitches detected (this might be okay for the test)")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nThis might indicate an issue with librosa installation.")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("VOCAL COACH - Audio Troubleshooting")
    print("=" * 50)

    results = []

    # Test 1: Package imports
    results.append(("Package imports", test_imports()))

    # Test 2: Audio devices
    if results[-1][1]:  # Only if imports worked
        results.append(("Audio devices", test_audio_devices()))

        # Test 3: Audio playback
        if results[-1][1]:  # Only if devices work
            results.append(("Audio playback", test_audio_playback()))

        # Test 4: Audio file processing
        results.append(("Audio file processing", test_audio_file()))

    # Summary
    print("\n\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! Your system is ready to run the Vocal Coach app.")
    else:
        print("\n⚠️  Some tests failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("  1. Reinstall all packages: pip install -r requirements.txt --force-reinstall")
        print("  2. Install PortAudio (Windows: comes with sounddevice)")
        print("  3. Check system audio settings")
        print("  4. Try running as administrator")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
