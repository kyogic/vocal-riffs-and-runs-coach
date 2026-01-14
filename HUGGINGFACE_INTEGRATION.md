# Integrating Hugging Face Models for Pitch Detection

This guide explains how to use pre-trained models or train custom models for improved pitch detection.

## Option 1: Use Pre-trained Models (RECOMMENDED - Easiest)

### Best Pre-trained Models:

1. **CREPE** (Convolutional Representation for Pitch Estimation)
   - State-of-the-art deep learning pitch detector
   - Much better than librosa's pyin/yin
   - Already available via pip

2. **Basic-Pitch** (Spotify's model)
   - Excellent for musical note transcription
   - Multi-pitch detection (polyphonic)
   - Easy to integrate

### Installing CREPE (Easiest Integration):

```bash
pip install crepe tensorflow
```

### Code Integration for CREPE:

Add to your algorithm dropdown in `main.py`:

```python
# In PitchDetector.__init__:
self.algorithm_combo.addItem('CREPE (Deep Learning)', 'crepe')

# In PitchDetector.detect_pitch method:
elif self.algorithm == 'crepe':
    import crepe

    # Resample to 16kHz (CREPE requirement)
    audio_16k = librosa.resample(audio, orig_sr=self.sr, target_sr=16000)

    # Run CREPE
    time, frequency, confidence, _ = crepe.predict(
        audio_16k,
        16000,
        viterbi=True,
        model_capacity='small'  # Options: tiny, small, medium, large, full
    )

    # Resample back to your hop_length
    target_frames = int(len(audio) / self.hop_length)
    f0 = np.interp(
        np.arange(target_frames) * self.hop_length / self.sr,
        time,
        frequency
    )

    voiced_probs = np.interp(
        np.arange(target_frames) * self.hop_length / self.sr,
        time,
        confidence
    )

    # Threshold for voiced/unvoiced
    f0[voiced_probs < 0.5] = np.nan
    voiced_flag = ~np.isnan(f0)

    return f0, voiced_flag, voiced_probs
```

### Installing Basic-Pitch:

```bash
pip install basic-pitch
```

### Code Integration for Basic-Pitch:

```python
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

def detect_pitch_basic_pitch(audio, sr):
    # Basic-Pitch expects numpy array
    model_output, midi_data, note_events = predict(
        audio,
        sr,
        ICASSP_2022_MODEL_PATH
    )

    # Extract pitch from note events
    # note_events format: [(start_time, end_time, pitch_midi, amplitude)]
    return note_events
```

---

## Option 2: Fine-tune a Pre-trained Model (Advanced)

If CREPE/Basic-Pitch don't work well for your specific use case, you can fine-tune.

### Requirements:
- **Dataset**: 100+ hours of vocal audio with ground truth pitch annotations
- **Compute**: GPU with 8GB+ VRAM
- **Time**: Several hours to days
- **Libraries**: PyTorch/TensorFlow, Hugging Face Transformers

### Dataset Preparation:

1. **Collect Audio**:
   ```
   vocal_dataset/
   ├── train/
   │   ├── audio_001.wav
   │   ├── audio_002.wav
   │   └── ...
   └── annotations/
       ├── audio_001.f0  (text file with frequency per frame)
       ├── audio_002.f0
       └── ...
   ```

2. **Annotation Format** (one frequency per line):
   ```
   0.0      # Unvoiced
   440.0    # A4
   493.88   # B4
   0.0      # Unvoiced
   ```

### Fine-tuning CREPE:

```python
import tensorflow as tf
from crepe import build_and_load_model
import numpy as np

# 1. Load pre-trained CREPE
model = build_and_load_model('small')

# 2. Freeze early layers (keep feature extraction)
for layer in model.layers[:-5]:
    layer.trainable = False

# 3. Prepare your dataset
def load_dataset(audio_dir, annotation_dir):
    X_train = []  # Audio samples
    y_train = []  # Ground truth frequencies

    for audio_file in os.listdir(audio_dir):
        audio, sr = librosa.load(os.path.join(audio_dir, audio_file))

        # Load corresponding annotation
        f0_file = audio_file.replace('.wav', '.f0')
        f0_true = np.loadtxt(os.path.join(annotation_dir, f0_file))

        X_train.append(audio)
        y_train.append(f0_true)

    return np.array(X_train), np.array(y_train)

X_train, y_train = load_dataset('vocal_dataset/train', 'vocal_dataset/annotations')

# 4. Fine-tune
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='mse'  # Mean squared error for frequency prediction
)

model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=10,
    validation_split=0.2
)

# 5. Save fine-tuned model
model.save('models/crepe_vocal_finetuned')
```

### Using Your Fine-tuned Model:

```python
# Load your fine-tuned model
model = tf.keras.models.load_model('models/crepe_vocal_finetuned')

# Use it in your app
def detect_pitch_custom(audio, sr):
    # Resample to 16kHz
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Run inference
    predictions = model.predict(audio_16k)

    return predictions
```

---

## Option 3: Train from Scratch (Very Advanced)

Only do this if pre-trained models fail completely.

### Architecture:

```python
import torch
import torch.nn as nn

class VocalPitchDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=512, stride=256),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=64, stride=8),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 256, kernel_size=16, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(256, 128, num_layers=2, bidirectional=True)

        # Output layer (predicts frequency in Hz)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: (batch, 1, samples)
        x = self.conv_layers(x)
        x = x.permute(2, 0, 1)  # (time, batch, features)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x.squeeze(-1)

# Training loop
model = VocalPitchDetector()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    for audio_batch, f0_batch in train_loader:
        optimizer.zero_grad()

        predictions = model(audio_batch)
        loss = criterion(predictions, f0_batch)

        loss.backward()
        optimizer.step()
```

---

## Recommendation: Start with CREPE

**Why CREPE is best for your app:**

1. ✓ **Already trained** on massive vocal datasets
2. ✓ **Easy to integrate** (pip install)
3. ✓ **Much more accurate** than pyin/yin
4. ✓ **No dataset needed**
5. ✓ **Works out of the box**

### Integration Steps:

1. Install CREPE:
   ```bash
   pip install crepe tensorflow
   ```

2. Add to your algorithm dropdown

3. Test on your a cappella tracks

4. Compare results with pyin/yin

5. If CREPE works better, make it default

---

## Comparison:

| Method | Accuracy | Speed | Setup | Dataset Needed |
|--------|----------|-------|-------|----------------|
| pYIN | 60-70% | Fast | None | No |
| YIN | 65-75% | Fast | None | No |
| **CREPE** | **90-95%** | Medium | pip install | **No** |
| Basic-Pitch | 85-90% | Medium | pip install | No |
| Fine-tuned | 95-98% | Medium | Complex | Yes (100+ hours) |
| From Scratch | 95-98% | Medium | Very Complex | Yes (500+ hours) |

---

## Next Steps:

1. **Try CREPE first** (easiest, best results)
2. **If CREPE is too slow**, try Basic-Pitch
3. **If results still poor**, collect dataset and fine-tune
4. **If you have domain expertise**, consider training from scratch

---

## Example: Adding CREPE to Your App

Create a new file `crepe_integration.py`:

```python
import numpy as np
import librosa
import crepe

def detect_pitch_crepe(audio, sr, hop_length=256):
    """
    Detect pitch using CREPE deep learning model

    Args:
        audio: Audio signal
        sr: Sample rate
        hop_length: Hop length for frame analysis

    Returns:
        f0, voiced_flag, voiced_probs: Same format as librosa.pyin
    """
    # Resample to 16kHz (CREPE requirement)
    target_sr = 16000
    if sr != target_sr:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    else:
        audio_16k = audio

    # Run CREPE
    time, frequency, confidence, _ = crepe.predict(
        audio_16k,
        target_sr,
        viterbi=True,
        model_capacity='small',  # Options: tiny, small, medium, large, full
        step_size=10  # milliseconds between predictions
    )

    # Resample to match your hop_length
    target_frames = int(len(audio) / hop_length)
    f0 = np.interp(
        np.arange(target_frames) * hop_length / sr,
        time,
        frequency
    )

    voiced_probs = np.interp(
        np.arange(target_frames) * hop_length / sr,
        time,
        confidence
    )

    # Threshold for voiced/unvoiced
    voiced_threshold = 0.5
    f0[voiced_probs < voiced_threshold] = np.nan
    voiced_flag = ~np.isnan(f0)

    return f0, voiced_flag, voiced_probs
```

Then in your `main.py`, import and use it:

```python
from crepe_integration import detect_pitch_crepe

# In PitchDetector.detect_pitch():
elif self.algorithm == 'crepe':
    return detect_pitch_crepe(audio, self.sr, self.hop_length)
```

---

## Resources:

- **CREPE Paper**: https://arxiv.org/abs/1802.06182
- **Basic-Pitch**: https://github.com/spotify/basic-pitch
- **Hugging Face Audio**: https://huggingface.co/docs/transformers/tasks/audio_classification
- **MIR Datasets**: https://github.com/mir-dataset-loaders/mirdata

---

## Questions?

Feel free to ask about:
- Specific integration issues
- Dataset preparation
- Model selection
- Performance optimization
- Accuracy improvements
