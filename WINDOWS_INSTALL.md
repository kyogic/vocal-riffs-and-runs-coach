# Windows Installation Guide

## Method 1: Automated Installation (Recommended)

1. **First, upgrade pip:**
   ```powershell
   python -m pip install --upgrade pip
   ```

2. **Create and activate virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   If you get an execution policy error:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Install all dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```powershell
   python main.py
   ```

## Method 2: Step-by-Step Installation (If Method 1 Fails)

If you encounter dependency resolution errors, install packages individually:

```powershell
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install packages one by one
pip install PyQt5
pip install numpy
pip install scipy
pip install matplotlib
pip install soundfile
pip install sounddevice
pip install librosa

# Run the application
python main.py
```

## Method 3: Using Command Prompt Instead of PowerShell

If PowerShell continues to cause issues:

1. Open **Command Prompt** (cmd.exe)
2. Navigate to the project directory:
   ```cmd
   cd E:\claude projects\vocal-riffs-and-runs-coach
   ```
3. Create and activate virtual environment:
   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```
4. Upgrade pip and install:
   ```cmd
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Run the application:
   ```cmd
   python main.py
   ```

## Troubleshooting

### Error: "Could not find a version that satisfies the requirement"
- **Solution**: Upgrade pip first: `python -m pip install --upgrade pip`
- Check your Python version: `python --version` (need Python 3.8 or higher)

### Error: "Microsoft Visual C++ 14.0 or greater is required"
- **Solution**: Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Or install pre-built wheels from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/)

### Error: Execution policy prevents script running
- **Solution**: Run in PowerShell as Administrator:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### Audio playback not working
- **Solution**: Install PortAudio:
  - Download from: http://www.portaudio.com/
  - Or install sounddevice with binary: `pip install sounddevice --prefer-binary`

## Verifying Installation

After installation, verify everything works:

```powershell
python -c "import PyQt5; print('PyQt5: OK')"
python -c "import numpy; print('numpy: OK')"
python -c "import librosa; print('librosa: OK')"
python -c "import soundfile; print('soundfile: OK')"
python -c "import matplotlib; print('matplotlib: OK')"
```

If all print "OK", you're ready to go!

## Quick Start After Installation

```powershell
# From the project directory
.\venv\Scripts\Activate.ps1  # or activate.bat in cmd
python main.py
```

## Deactivating Virtual Environment

When you're done:
```powershell
deactivate
```
