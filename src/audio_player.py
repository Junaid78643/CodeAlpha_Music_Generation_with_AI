"""
Audio Synthesizer & Player Module
Converts musical note/chord sequences directly into:
  1. Standard MIDI files (.mid)
  2. High-fidelity acoustic WAV audio files (.wav) using additive harmonic synthesis and ADSR envelopes
"""

import math
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from music21 import pitch, note, chord, stream, midi, tempo, meter

BASE_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = BASE_DIR / "data" / "generated"

NOTE_NAME_MAP = {
    'C': 0, 'C#': 1, 'D-': 1, 'D': 2, 'D#': 3, 'E-': 3,
    'E': 4, 'F': 5, 'F#': 6, 'G-': 6, 'G': 7, 'G#': 8,
    'A-': 8, 'A': 9, 'A#': 10, 'B-': 10, 'B': 11
}

def note_to_freq(note_str):
    """
    Converts note pitch name (e.g. 'C4', 'G#3', 'Eb5') to Hertz frequency.
    """
    try:
        p = pitch.Pitch(note_str)
        return float(p.frequency)
    except Exception:
        # Fallback manual calculation
        try:
            octave = int(note_str[-1])
            name = note_str[:-1]
            semitone = NOTE_NAME_MAP.get(name, 0)
            midi_num = 12 + octave * 12 + semitone
            return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))
        except Exception:
            return 440.0

def generate_adsr_envelope(total_samples, attack=0.03, decay=0.1, sustain_level=0.6, release=0.15, sr=44100):
    """
    Generates an ADSR (Attack, Decay, Sustain, Release) amplitude envelope.
    """
    a_samples = int(attack * sr)
    d_samples = int(decay * sr)
    r_samples = int(release * sr)
    s_samples = max(0, total_samples - (a_samples + d_samples + r_samples))

    # Re-normalize if duration is very short
    if a_samples + d_samples + r_samples > total_samples:
        a_samples = int(total_samples * 0.15)
        d_samples = int(total_samples * 0.2)
        r_samples = int(total_samples * 0.3)
        s_samples = total_samples - (a_samples + d_samples + r_samples)

    attack_curve = np.linspace(0.0, 1.0, a_samples)
    decay_curve = np.linspace(1.0, sustain_level, d_samples)
    sustain_curve = np.full(s_samples, sustain_level)
    release_curve = np.linspace(sustain_level, 0.0, r_samples)

    env = np.concatenate([attack_curve, decay_curve, sustain_curve, release_curve])
    if len(env) < total_samples:
        env = np.pad(env, (0, total_samples - len(env)))
    else:
        env = env[:total_samples]
    return env

def synthesize_acoustic_tone(freqs, duration_sec, sample_rate=44100):
    """
    Synthesizes rich acoustic piano/electric piano timbre with natural harmonics.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    wave = np.zeros_like(t)

    for f in freqs:
        if f <= 0:
            continue
        # Fundamental + rich physical harmonics
        harmonics = [
            (1.0, 1.0),      # Fundamental
            (2.0, 0.55),     # 2nd harmonic (octave)
            (3.0, 0.30),     # 3rd harmonic (perfect fifth)
            (4.0, 0.15),     # 4th harmonic
            (5.0, 0.08),     # 5th harmonic (major third)
            (6.0, 0.04)      # 6th harmonic
        ]
        tone = np.zeros_like(t)
        for mult, amp in harmonics:
            if mult * f < sample_rate / 2: # Anti-aliasing check
                tone += amp * np.sin(2 * np.pi * mult * f * t)
        wave += tone

    if len(freqs) > 0:
        wave /= len(freqs)

    # Apply ADSR envelope
    envelope = generate_adsr_envelope(len(wave), sr=sample_rate)
    return wave * envelope

def sequence_to_midi_and_audio(sequence_tokens, filename_prefix="generated_music", bpm=115, sample_rate=44100):
    """
    Converts list of note tokens (e.g. ['C4:1.0', 'E4.G4:0.5', 'rest:0.5'])
    into both a standard MIDI file (.mid) and a high quality WAV audio file (.wav).
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    midi_path = GENERATED_DIR / f"{filename_prefix}.mid"
    wav_path = GENERATED_DIR / f"{filename_prefix}.wav"

    s = stream.Score()
    p = stream.Part()
    p.append(meter.TimeSignature('4/4'))
    p.append(tempo.MetronomeMark(number=bpm))

    audio_segments = []
    seconds_per_quarter = 60.0 / bpm

    for token in sequence_tokens:
        if ":" in token:
            pitch_part, dur_part = token.split(":", 1)
            try:
                quarter_len = float(dur_part)
            except ValueError:
                quarter_len = 1.0
        else:
            pitch_part = token
            quarter_len = 1.0

        dur_sec = quarter_len * seconds_per_quarter

        if pitch_part == "rest":
            r = note.Rest()
            r.quarterLength = quarter_len
            p.append(r)
            # Silence in audio
            silence = np.zeros(int(sample_rate * dur_sec))
            audio_segments.append(silence)

        elif "." in pitch_part:
            # Chord
            pitches = pitch_part.split(".")
            c = chord.Chord(pitches)
            c.quarterLength = quarter_len
            p.append(c)

            freqs = [note_to_freq(pt) for pt in pitches]
            tone = synthesize_acoustic_tone(freqs, dur_sec, sample_rate)
            audio_segments.append(tone)

        else:
            # Single note
            n = note.Note(pitch_part)
            n.quarterLength = quarter_len
            p.append(n)

            freq = note_to_freq(pitch_part)
            tone = synthesize_acoustic_tone([freq], dur_sec, sample_rate)
            audio_segments.append(tone)

    s.append(p)

    # 1. Save standard MIDI file
    mf = midi.translate.music21ObjectToMidiFile(s)
    mf.open(str(midi_path), 'wb')
    mf.write()
    mf.close()
    print(f"[OK] Exported MIDI to: {midi_path}")

    # 2. Render and save WAV audio
    if audio_segments:
        full_audio = np.concatenate(audio_segments)
        # Normalize to -1.0 to +1.0
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = (full_audio / max_val) * 0.95
        # Convert to 16-bit PCM integer
        audio_int16 = (full_audio * 32767).astype(np.int16)
        wavfile.write(str(wav_path), sample_rate, audio_int16)
        print(f"[OK] Exported WAV Audio to: {wav_path} ({len(full_audio)/sample_rate:.2f} seconds)")

    return str(midi_path), str(wav_path)

if __name__ == "__main__":
    test_seq = ["C4:1.0", "E4.G4:1.0", "rest:0.5", "A4:0.5", "G4:1.0", "C4.E4.G4.C5:2.0"]
    m, w = sequence_to_midi_and_audio(test_seq, "test_render")
