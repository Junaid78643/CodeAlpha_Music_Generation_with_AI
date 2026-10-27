"""
Music Generation Engine
Uses the trained LSTM model to autoregressively compose new music sequences,
with configurable temperature, length, and genre styling.
Exports to standard MIDI and high-fidelity WAV audio.
"""

import os
import json
import random
import argparse
import numpy as np
from pathlib import Path
from src.model import NumpyMusicLSTM, HAS_TORCH
from src.audio_player import sequence_to_midi_and_audio

if HAS_TORCH:
    import torch
    from src.model import TorchMusicLSTM

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
GENERATED_DIR = BASE_DIR / "data" / "generated"

class MusicGenerator:
    def __init__(self, genre="classical", engine="numpy"):
        self.genre = genre
        self.engine = engine
        self.data_path = DATA_DIR / f"{genre}_processed.json"

        if not self.data_path.exists():
            raise FileNotFoundError(f"Processed dataset {self.data_path} not found.")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.vocab = self.metadata["vocab"]
        self.inv_vocab = {int(idx): tok for tok, idx in self.vocab.items()}
        self.vocab_size = self.metadata["vocab_size"]
        self.sequence_length = self.metadata["sequence_length"]
        self.all_inputs = self.metadata["inputs"]

        self.model = self._load_model()

    def _load_model(self):
        """
        Loads the trained model weights.
        """
        numpy_weight_path = MODELS_DIR / f"{self.genre}_lstm_numpy.npz"
        torch_weight_path = MODELS_DIR / f"{self.genre}_lstm_torch.pt"

        if self.engine == "torch" and HAS_TORCH and torch_weight_path.exists():
            print(f"[*] Loading PyTorch LSTM model from {torch_weight_path}...")
            model = TorchMusicLSTM(self.vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
            model.load_state_dict(torch.load(torch_weight_path, map_location="cpu"))
            model.eval()
            return model
        elif numpy_weight_path.exists():
            print(f"[*] Loading NumPy LSTM model from {numpy_weight_path}...")
            return NumpyMusicLSTM.load_weights(str(numpy_weight_path))
        else:
            print(f"[!] Pre-trained weights not found for {self.genre}. Initializing fresh model...")
            return NumpyMusicLSTM(self.vocab_size, embedding_dim=64, hidden_dim=128)

    def sample_with_temperature(self, probabilities, temperature=1.0, top_k=8):
        """
        Applies temperature scaling and top-k filtering before sampling next token.
        """
        probs = np.copy(probabilities)
        if temperature != 1.0:
            # Re-scale logits safely
            logits = np.log(np.maximum(probs, 1e-12)) / max(temperature, 0.1)
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

        # Top-k filtering to prevent rare off-key artifacts
        if top_k and top_k < len(probs):
            top_k_indices = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs, dtype=bool)
            mask[top_k_indices] = True
            probs[~mask] = 0.0
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                probs /= prob_sum
            else:
                probs = np.ones_like(probs) / len(probs)

        return np.random.choice(len(probs), p=probs)

    def generate(self, num_notes=48, temperature=0.9, seed_tokens=None, filename="ai_composition"):
        """
        Autoregressively generates a novel sequence of musical tokens.
        """
        # 1. Establish seed sequence
        if seed_tokens and len(seed_tokens) >= self.sequence_length:
            curr_sequence = [self.vocab.get(t, 0) for t in seed_tokens[-self.sequence_length:]]
        else:
            # Pick a natural musical opening motif from the dataset
            rand_idx = random.randint(0, len(self.all_inputs) - 1)
            curr_sequence = list(self.all_inputs[rand_idx])

        generated_tokens = [self.inv_vocab.get(idx, "C4:1.0") for idx in curr_sequence]
        print(f"[*] Generating {num_notes} new notes/chords with Temperature = {temperature:.2f}...")

        # 2. Autoregressive loop
        for step in range(num_notes):
            window = curr_sequence[-self.sequence_length:]

            if isinstance(self.model, NumpyMusicLSTM):
                probs = self.model.predict(window, temperature=temperature)
            else:
                # PyTorch
                with torch.no_grad():
                    input_tensor = torch.tensor([window], dtype=torch.long)
                    logits, _ = self.model(input_tensor)
                    probs = torch.softmax(logits / max(temperature, 0.1), dim=-1).numpy().flatten()

            next_idx = self.sample_with_temperature(probs, temperature=temperature, top_k=10)
            next_token = self.inv_vocab.get(next_idx, "C4:1.0")

            generated_tokens.append(next_token)
            curr_sequence.append(next_idx)

        # 3. Export to MIDI & WAV
        bpm = 100 if self.genre == "classical" else 118
        midi_path, wav_path = sequence_to_midi_and_audio(
            generated_tokens,
            filename_prefix=f"{filename}_{self.genre}",
            bpm=bpm
        )

        return {
            "genre": self.genre,
            "num_notes": len(generated_tokens),
            "tokens": generated_tokens,
            "midi_path": midi_path,
            "wav_path": wav_path,
            "temperature": temperature
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", type=str, default="classical", choices=["classical", "jazz"])
    parser.add_argument("--notes", type=int, default=40)
    parser.add_argument("--temp", type=float, default=0.85)
    args = parser.parse_args()

    gen = MusicGenerator(genre=args.genre)
    result = gen.generate(num_notes=args.notes, temperature=args.temp)
    print(f"\n[OK] Generated {len(result['tokens'])} notes!")
    print(f"     MIDI: {result['midi_path']}")
    print(f"     WAV:  {result['wav_path']}")
