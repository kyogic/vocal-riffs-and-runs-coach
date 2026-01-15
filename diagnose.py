#!/usr/bin/env python3
"""
Diagnostic script to help identify why the app won't start
"""

import sys
import traceback

print("=" * 70)
print("VOCAL COACH APP DIAGNOSTIC")
print("=" * 70)

# Step 1: Check Python version
print("\n1. Checking Python version...")
print(f"   Python {sys.version}")
if sys.version_info < (3, 7):
    print("   ⚠️  WARNING: Python 3.7+ is recommended")
else:
    print("   ✓ Python version OK")

# Step 2: Check required modules
print("\n2. Checking required modules...")
required_modules = [
    'numpy',
    'librosa',
    'scipy',
    'PyQt5',
    'matplotlib',
    'sounddevice'
]

missing_modules = []
for module in required_modules:
    try:
        __import__(module)
        print(f"   ✓ {module}")
    except ImportError:
        print(f"   ✗ {module} - NOT FOUND")
        missing_modules.append(module)

if missing_modules:
    print(f"\n   ⚠️  MISSING MODULES: {', '.join(missing_modules)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Step 3: Try importing the main module
print("\n3. Importing main module...")
try:
    import main
    print("   ✓ main.py imported successfully")
except Exception as e:
    print(f"   ✗ IMPORT ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Try creating QApplication
print("\n4. Creating QApplication...")
try:
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    print("   ✓ QApplication created")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

# Step 5: Try creating the main window
print("\n5. Creating VocalCoachApp window...")
try:
    window = main.VocalCoachApp()
    print("   ✓ VocalCoachApp created successfully")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

# Step 6: Try showing the window
print("\n6. Attempting to show window...")
try:
    window.show()
    print("   ✓ Window shown successfully")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL DIAGNOSTICS PASSED")
print("=" * 70)
print("\nThe app should be running now. If you see the window, everything is working!")
print("Press Ctrl+C to exit this diagnostic script.")
print("=" * 70)

# Keep the app running
sys.exit(app.exec_())
