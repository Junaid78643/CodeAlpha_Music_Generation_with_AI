"""
Preprocessing Module for AI Music Generation
Parses MIDI files using music21 & mido into note/chord/duration tokens,
builds token vocabulary, and generates input-output sequence pairs for LSTM training.
"""

import os
import json
import glob
from pathlib import Path
from music21 import converter, instrument, note, chord

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

class MidiPreprocessor:
    def __init__(self, sequence_length=32, include_duration=True):
        self.sequence_length = sequence_length
        self.include_duration = include_duration
        self.vocab = {}
        self.inv_vocab = {}
        self.vocab_size = 0
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def parse_midi_file(self, file_path):
        """
        Parses a single MIDI file into a sequence of token strings.
        Tokens represent:
          - Notes (e.g. 'C4:1.0')
          - Chords (e.g. 'C4.E4.G4:1.0')
          - Rests (e.g. 'rest:0.5')
        """
        tokens = []
        try:
            midi_score = converter.parse(file_path)
            if hasattr(midi_score, 'flatten'):
                notes_to_parse = midi_score.flatten().notesAndRests
            elif hasattr(midi_score, 'flat'):
                notes_to_parse = midi_score.flat.notesAndRests
            else:
                notes_to_parse = midi_score.recurse()

            for element in notes_to_parse:
                # Quantize duration to musical fractions (e.g., 0.25, 0.5, 1.0, 2.0)
                dur = round(float(element.quarterLength), 2)
                if dur <= 0:
                    dur = 0.25

                if isinstance(element, note.Note):
                    pitch_str = str(element.pitch)
                    token = f"{pitch_str}:{dur}" if self.include_duration else pitch_str
                    tokens.append(token)

                elif isinstance(element, chord.Chord):
                    pitches = sorted([str(p) for p in element.pitches])
                    chord_str = ".".join(pitches)
                    token = f"{chord_str}:{dur}" if self.include_duration else chord_str
                    tokens.append(token)

                elif isinstance(element, note.Rest):
                    # Keep major rests
                    if dur >= 0.5:
                        token = f"rest:{dur}" if self.include_duration else "rest"
                        tokens.append(token)

        except Exception as e:
            print(f"  [!] Warning: Failed to parse {Path(file_path).name}: {e}")
        return tokens

    def process_directory(self, midi_folder):
        """
        Scans a directory of MIDI files and extracts complete token sequences.
        """
        midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + \
                     glob.glob(os.path.join(midi_folder, "*.midi"))
        print(f"[*] Processing {len(midi_files)} MIDI files in {midi_folder}...")

        all_tokens = []
        file_token_map = {}

        for fpath in midi_files:
            file_name = Path(fpath).name
            file_tokens = self.parse_midi_file(fpath)
            if len(file_tokens) >= self.sequence_length:
                all_tokens.extend(file_tokens)
                file_token_map[file_name] = file_tokens
                print(f"  [+] Parsed {file_name}: {len(file_tokens)} tokens")

        print(f"[OK] Extracted total of {len(all_tokens)} musical tokens across {len(file_token_map)} pieces.")
        return all_tokens, file_token_map

    def build_vocabulary(self, tokens):
        """
        Builds bijective mapping between tokens and numeric integer IDs.
        """
        unique_tokens = sorted(list(set(tokens)))
        self.vocab = {tok: idx for idx, tok in enumerate(unique_tokens)}
        self.inv_vocab = {idx: tok for idx, tok in enumerate(unique_tokens)}
        self.vocab_size = len(unique_tokens)
        print(f"[OK] Built vocabulary with {self.vocab_size} unique musical tokens.")
        return self.vocab, self.inv_vocab

    def create_dataset_sequences(self, tokens):
        """
        Generates sliding window training samples:
        Input: sequence of length N
        Target: next token (N+1)
        """
        if not self.vocab:
            self.build_vocabulary(tokens)

        network_input = []
        network_output = []

        for i in range(len(tokens) - self.sequence_length):
            seq_in = tokens[i : i + self.sequence_length]
            seq_out = tokens[i + self.sequence_length]

            network_input.append([self.vocab[tok] for tok in seq_in])
            network_output.append(self.vocab[seq_out])

        print(f"[OK] Created {len(network_input)} training sequence pairs (window size: {self.sequence_length}).")
        return network_input, network_output

    def save_processed_data(self, genre, tokens, network_input, network_output):
        """
        Persists preprocessed sequences, vocabulary, and metadata to JSON.
        """
        save_path = PROCESSED_DIR / f"{genre}_processed.json"
        data = {
            "genre": genre,
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "tokens_sample": tokens[:200],
            "total_tokens": len(tokens),
            "num_sequences": len(network_input),
            "inputs": network_input,
            "outputs": network_output
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"[OK] Saved processed dataset to {save_path}")
        return str(save_path)

    def load_processed_data(self, genre):
        """
        Loads cached preprocessed data and vocabulary.
        """
        load_path = PROCESSED_DIR / f"{genre}_processed.json"
        if not load_path.exists():
            raise FileNotFoundError(f"Processed data not found at {load_path}")
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.inv_vocab = {int(v): k for k, v in self.vocab.items()}
        self.vocab_size = data["vocab_size"]
        self.sequence_length = data["sequence_length"]
        return data

if __name__ == "__main__":
    preprocessor = MidiPreprocessor(sequence_length=32)
    # Test on classical folder
    classical_dir = BASE_DIR / "data" / "midi_dataset" / "classical"
    if classical_dir.exists():
        tokens, _ = preprocessor.process_directory(str(classical_dir))
        if tokens:
            X, y = preprocessor.create_dataset_sequences(tokens)
            preprocessor.save_processed_data("classical", tokens, X, y)
