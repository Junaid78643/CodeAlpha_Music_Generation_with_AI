"""
AI Music Generation Web Studio - Flask Backend
Provides an interactive API and UI for:
  - Generating new classical and jazz music using trained LSTM models
  - Real-time Piano Roll visualization data
  - Streaming synthesized audio (.wav) and downloading standard MIDI (.mid)
  - Inspecting dataset tokens and training loss curves
"""

import os
import glob
import json
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from music21 import pitch

from src.generate import MusicGenerator
from src.train import train_numpy_lstm

app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = DATA_DIR / "generated"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
MIDI_DATASET_DIR = DATA_DIR / "midi_dataset"

GENERATED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def token_to_piano_roll_events(tokens, bpm=115):
    """
    Converts note and chord tokens into absolute timestamped events
    for the HTML5 Canvas Piano Roll visualizer.
    """
    events = []
    current_time = 0.0
    seconds_per_quarter = 60.0 / bpm

    for idx, tok in enumerate(tokens):
        if ":" in tok:
            p_part, dur_part = tok.split(":", 1)
            try:
                quarter_len = float(dur_part)
            except ValueError:
                quarter_len = 1.0
        else:
            p_part = tok
            quarter_len = 1.0

        dur_sec = quarter_len * seconds_per_quarter

        if p_part == "rest":
            events.append({
                "type": "rest",
                "start": round(current_time, 3),
                "duration": round(dur_sec, 3),
                "pitch": "Rest",
                "midi": 0
            })
        elif "." in p_part:
            pitches = p_part.split(".")
            for p_str in pitches:
                try:
                    p_obj = pitch.Pitch(p_str)
                    midi_num = p_obj.midi
                except Exception:
                    midi_num = 60
                events.append({
                    "type": "chord_note",
                    "start": round(current_time, 3),
                    "duration": round(dur_sec, 3),
                    "pitch": p_str,
                    "midi": midi_num
                })
        else:
            try:
                p_obj = pitch.Pitch(p_part)
                midi_num = p_obj.midi
            except Exception:
                midi_num = 60
            events.append({
                "type": "note",
                "start": round(current_time, 3),
                "duration": round(dur_sec, 3),
                "pitch": p_part,
                "midi": midi_num
            })

        current_time += dur_sec

    return events, round(current_time, 2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    Returns statistics about dataset, models, and system status.
    """
    classical_files = len(glob.glob(str(MIDI_DATASET_DIR / "classical" / "*.mid")))
    jazz_files = len(glob.glob(str(MIDI_DATASET_DIR / "jazz" / "*.mid")))

    classical_ready = (MODELS_DIR / "classical_lstm_numpy.npz").exists()
    jazz_ready = (MODELS_DIR / "jazz_lstm_numpy.npz").exists()

    classical_tokens = 0
    jazz_tokens = 0
    classical_vocab = 0
    jazz_vocab = 0

    c_proc = PROCESSED_DIR / "classical_processed.json"
    if c_proc.exists():
        with open(c_proc, "r", encoding="utf-8") as f:
            d = json.load(f)
            classical_tokens = d.get("total_tokens", 0)
            classical_vocab = d.get("vocab_size", 0)

    j_proc = PROCESSED_DIR / "jazz_processed.json"
    if j_proc.exists():
        with open(j_proc, "r", encoding="utf-8") as f:
            d = json.load(f)
            jazz_tokens = d.get("total_tokens", 0)
            jazz_vocab = d.get("vocab_size", 0)

    return jsonify({
        "classical": {
            "midi_files": classical_files,
            "tokens": classical_tokens,
            "vocab_size": classical_vocab,
            "model_trained": classical_ready
        },
        "jazz": {
            "midi_files": jazz_files,
            "tokens": jazz_tokens,
            "vocab_size": jazz_vocab,
            "model_trained": jazz_ready
        }
    })

@app.route("/api/history/<genre>", methods=["GET"])
def get_history(genre):
    """
    Returns training loss history for the specified genre.
    """
    history_file = MODELS_DIR / f"{genre}_history.json"
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"epoch": [], "loss": [], "time": []})

@app.route("/api/generate", methods=["POST"])
def generate_music():
    """
    Generates music sequence using trained AI model.
    """
    data = request.get_json() or {}
    genre = data.get("genre", "classical")
    temperature = float(data.get("temperature", 0.85))
    num_notes = int(data.get("num_notes", 45))
    bpm = int(data.get("bpm", 110 if genre == "jazz" else 100))

    try:
        gen = MusicGenerator(genre=genre)
        timestamp = int(time.time())
        filename = f"{genre}_ai_{timestamp}"
        result = gen.generate(
            num_notes=num_notes,
            temperature=temperature,
            filename=filename
        )

        tokens = result["tokens"]
        events, total_duration = token_to_piano_roll_events(tokens, bpm=bpm)

        midi_filename = f"{filename}_{genre}.mid"
        wav_filename = f"{filename}_{genre}.wav"

        return jsonify({
            "success": True,
            "genre": genre,
            "temperature": temperature,
            "num_notes": len(tokens),
            "tokens": tokens,
            "events": events,
            "total_duration": total_duration,
            "midi_url": f"/media/generated/{midi_filename}",
            "wav_url": f"/media/generated/{wav_filename}",
            "bpm": bpm
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def train_model():
    """
    Trains the LSTM model for a specific genre on-demand.
    """
    data = request.get_json() or {}
    genre = data.get("genre", "classical")
    epochs = int(data.get("epochs", 15))
    samples = int(data.get("samples", 450))

    try:
        model, history = train_numpy_lstm(genre=genre, epochs=epochs, max_samples=samples)
        return jsonify({
            "success": True,
            "genre": genre,
            "epochs": epochs,
            "final_loss": history["loss"][-1] if history["loss"] else 0,
            "history": history
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/media/generated/<path:filename>")
def serve_generated(filename):
    return send_from_directory(str(GENERATED_DIR), filename, as_attachment=False)

@app.route("/download/<path:filename>")
def download_generated(filename):
    return send_from_directory(str(GENERATED_DIR), filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
