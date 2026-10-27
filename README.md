# AuraTone AI: Music Generation with Deep Learning

A deep learning music composition system and interactive web studio capable of learning musical styles (Classical & Jazz/Blues) from MIDI data and generating original, polyphonic musical sequences with real-time piano roll visualization, standard MIDI export, and high-fidelity 44.1 kHz acoustic WAV synthesis.

---

## 🌟 Key Features

- **Multi-Genre MIDI Dataset Pipeline**:
  - **Classical**: 30 authentic Bach, Beethoven, and Mozart chorales and inventions extracted via `music21.corpus` (9,093 musical tokens, 259 unique vocabulary classes).
  - **Jazz & Blues**: Scott Joplin Ragtime pieces and algorithmic ii-V-I jazz cadences with swing syncopations and 12-bar blues progressions (1,465 musical tokens, 178 unique vocabulary classes).
- **Intelligent Preprocessing (`music21` & `mido`)**:
  - Encodes single notes (`C4:1.0`), polyphonic chords (`C4.E4.G4:1.5`), and rhythmic rests (`rest:0.5`).
  - Creates sliding-window sequential training pairs $(X, y)$ with quantized musical durations.
- **Deep Recurrent Neural Network (LSTM) Architecture**:
  - Vectorized multi-gate LSTM cell (Input Gate, Forget Gate, Cell Gate, Output Gate) with Xavier/He initialization and Adam optimization.
  - Cross-entropy loss with gradient clipping and BPTT (Backpropagation Through Time).
  - Temperature-scaled autoregressive generation ($T \in [0.2, 1.8]$) with nucleus/top-$k$ filtering to prevent repetitive harmonic loops.
  - Dual-engine architecture: Native NumPy LSTM engine + PyTorch Deep LSTM (`TorchMusicLSTM`).
- **Physical Harmonic Audio Synthesizer**:
  - Additive acoustic synthesizer calculating precise fundamental and harmonic overtones ($1f, 2f, 3f, 4f, 5f, 6f$).
  - Natural ADSR amplitude envelope (Attack, Decay, Sustain, Release) simulating acoustic piano/electric keyboard dynamics.
  - Outputs both standard **MIDI (`.mid`)** and CD-quality **44.1 kHz WAV (`.wav`)**.
- **Interactive Web Studio**:
  - Real-time HTML5 Canvas **Piano Roll Visualizer** with animated playhead synchronized with audio playback.
  - Interactive controls: Genre selector, Creativity/Temperature slider, sequence length, and playback BPM.
  - Live model training panel with real-time Loss Convergence Curve.
  - One-click downloads for generated MIDI and WAV files.

---

## 📁 Project Structure

```
Task3_Music Generation with AI/
├── app.py                      # Flask Web Studio Backend & REST APIs
├── data/
│   ├── dataset_collector.py   # Extracts & generates Classical & Jazz MIDI datasets
│   ├── midi_dataset/          # Authentic MIDI datasets (.mid)
│   │   ├── classical/         # 30 Classical MIDI pieces (Bach, Beethoven)
│   │   └── jazz/              # 7 Jazz, Blues & Joplin Ragtime pieces
│   ├── processed/             # Tokenized sequence datasets (JSON)
│   └── generated/             # AI generated compositions (.mid & .wav)
├── models/                     # Trained neural network weights (.npz) & loss logs
│   ├── classical_lstm_numpy.npz
│   ├── jazz_lstm_numpy.npz
│   ├── classical_history.json
│   └── jazz_history.json
├── src/
│   ├── audio_player.py        # Synthesizer: Converts note sequences to MIDI & WAV
│   ├── generate.py            # Autoregressive generation engine with temperature sampling
│   ├── model.py               # Deep LSTM architecture (NumPy + PyTorch)
│   ├── preprocessing.py       # music21 & mido tokenization and vocabulary builder
│   └── train.py               # Model training script with loss logging
├── static/
│   ├── app.js                 # Piano roll canvas animation & audio synchronization
│   └── style.css              # Dark glassmorphism styling & glowing accents
├── templates/
│   └── index.html             # Web Studio layout
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Launch the Interactive Web Studio

Start the local web server:

```powershell
python app.py
```

Then open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

In the Web Studio you can:
1. Select between **Classical** or **Jazz & Blues**.
2. Adjust the **Creativity / Temperature** slider ($0.2$ for strict adherence, $0.85$ for sweet-spot, $1.4+$ for wild improvisation).
3. Set the desired **Sequence Length** (e.g. 45 notes) and **Playback Tempo** (BPM).
4. Click **Generate AI Composition** to watch the piano roll render and listen to your newly generated piece!
5. Download the master **MIDI (`.mid`)** or **WAV (`.wav`)** directly.

---

### 2. Command-Line Interface (CLI)

#### **Generate Music via CLI**
```powershell
# Generate Classical music
python -m src.generate --genre classical --notes 45 --temp 0.85

# Generate Jazz/Blues music
python -m src.generate --genre jazz --notes 40 --temp 1.1
```

Generated files are automatically saved to `data/generated/`.

#### **Train / Retrain Models**
```powershell
# Train on classical dataset
python -u -m src.train --genre classical --epochs 20 --samples 500

# Train on jazz dataset
python -u -m src.train --genre jazz --epochs 20 --samples 450
```

#### **Collect & Preprocess Datasets**
```powershell
# Extract MIDI dataset
python data/dataset_collector.py

# Preprocess token sequences
python src/preprocessing.py
```

---

## 📊 Training & Convergence Results

| Model Genre | Sequence Samples | Vocab Size | Epochs | Initial Loss | Final Loss | Output Formats |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical** | 500 | 259 | 20 | 5.0211 | **0.0006** | `.mid`, `.wav` |
| **Jazz & Blues** | 450 | 178 | 20 | 4.9440 | **0.0265** | `.mid`, `.wav` |

---

## 🎼 Technical Workflow Overview

```
[MIDI Dataset (Bach, Joplin)]
             │
             ▼
[music21 & mido Preprocessor] ──► Extracts Notes, Chords, Rests & Durations
             │
             ▼
[Token Vocabulary & Sliding Window (16-32 steps)]
             │
             ▼
[Deep LSTM Network] ──► BPTT + Adam Optimization ──► Loss: ~0.0006
             │
             ▼
[Temperature-Controlled Autoregressive Sampler]
             │
             ▼
┌────────────────────────────┴────────────────────────────┐
▼                                                         ▼
[music21 Stream to MIDI (.mid)]       [Additive Harmonic Synthesizer to WAV (.wav)]
```
