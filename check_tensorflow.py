"""
Diagnostic script to check TensorFlow installation and troubleshoot CREPE issues
"""
import sys
import struct
import platform

print("=" * 70)
print("TENSORFLOW DIAGNOSTICS")
print("=" * 70)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python architecture: {struct.calcsize('P') * 8}-bit")
print(f"Platform: {platform.platform()}")
print()

# Check if we can import TensorFlow at all
print("Step 1: Checking TensorFlow import...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow imported successfully!")
    print(f"  Version: {tf.__version__}")
    print(f"  Location: {tf.__file__}")
    print()

    # Try to access the problematic internal module
    print("Step 2: Checking TensorFlow internals...")
    try:
        from tensorflow.python import pywrap_tensorflow
        print("✓ TensorFlow internals loaded successfully!")
        print()

        # Check GPU/CPU support
        print("Step 3: Checking device support...")
        print(f"  GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
        print(f"  CPUs available: {len(tf.config.list_physical_devices('CPU'))}")
        print()

        # Try CREPE
        print("Step 4: Checking CREPE...")
        try:
            import crepe
            print(f"✓ CREPE imported successfully!")
            print(f"  Version: {crepe.__version__ if hasattr(crepe, '__version__') else 'unknown'}")
            print()
            print("=" * 70)
            print("✓✓✓ ALL CHECKS PASSED! CREPE should work fine.")
            print("=" * 70)
        except ImportError as e:
            print(f"✗ CREPE import failed: {e}")
            print()
            print("Solution: Install CREPE")
            print(f"  {sys.executable} -m pip install crepe")

    except ImportError as e:
        print(f"✗ TensorFlow internals failed: {e}")
        print()
        print("This is the DLL error. Let's fix it...")
        print()

except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    print()

# Provide solutions
print()
print("=" * 70)
print("RECOMMENDED SOLUTIONS (try in order):")
print("=" * 70)
print()
print("Solution 1: Reinstall TensorFlow (clean install)")
print("-" * 70)
print(f"{sys.executable} -m pip uninstall tensorflow tensorflow-intel -y")
print(f"{sys.executable} -m pip cache purge")
print(f"{sys.executable} -m pip install tensorflow==2.10.0")
print()
print("Solution 2: Try TensorFlow CPU-only version")
print("-" * 70)
print(f"{sys.executable} -m pip uninstall tensorflow tensorflow-intel -y")
print(f"{sys.executable} -m pip install tensorflow-cpu==2.10.0")
print()
print("Solution 3: Try latest TensorFlow")
print("-" * 70)
print(f"{sys.executable} -m pip uninstall tensorflow tensorflow-intel -y")
print(f"{sys.executable} -m pip install tensorflow")
print()
print("After trying any solution, run this script again to verify.")
print("=" * 70)
