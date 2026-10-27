"""
Model Training Script for Music Generation
Trains the LSTM Recurrent Neural Network on preprocessed musical token sequences.
Supports both pure NumPy LSTM engine and PyTorch Deep LSTM engine.
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from src.model import NumpyMusicLSTM, HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_numpy_lstm(genre="classical", epochs=40, batch_size=32, lr=0.005, max_samples=2500):
    """
    Trains NumpyMusicLSTM using BPTT and Adam optimization.
    """
    data_path = DATA_DIR / f"{genre}_processed.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset for {genre} not found at {data_path}. Run preprocessing first.")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["vocab"]
    vocab_size = data["vocab_size"]
    inputs = data["inputs"]
    outputs = data["outputs"]

    print(f"\n========================================================", flush=True)
    print(f"[*] Training NumPy LSTM Model on '{genre}' dataset", flush=True)
    print(f"[*] Total sequences: {len(inputs)}, Vocab size: {vocab_size}", flush=True)
    print(f"[*] Epochs: {epochs}, Learning Rate: {lr}", flush=True)
    print(f"========================================================\n", flush=True)

    # Subsample rich sequence pairs for swift convergence
    total = len(inputs)
    effective_samples = min(total, max_samples if max_samples else 450)
    indices = np.random.choice(total, effective_samples, replace=False)
    sub_inputs = [inputs[i][-16:] for i in indices] # Truncated 16-step BPTT window
    sub_outputs = [outputs[i] for i in indices]
    print(f"[*] Using {effective_samples} musical sequence samples with 16-step BPTT window.\n", flush=True)

    model = NumpyMusicLSTM(
        vocab_size=vocab_size,
        embedding_dim=48,
        hidden_dim=96,
        learning_rate=lr
    )

    history = {"epoch": [], "loss": [], "time": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        perm = np.random.permutation(effective_samples)
        total_loss = 0.0

        for idx in perm:
            x_seq = sub_inputs[idx]
            y_token = sub_outputs[idx]
            step_loss = model.train_step(x_seq, y_token)
            total_loss += step_loss

        avg_loss = total_loss / effective_samples
        epoch_dur = time.time() - epoch_start
        history["epoch"].append(epoch)
        history["loss"].append(round(float(avg_loss), 4))
        history["time"].append(round(epoch_dur, 2))

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  [Epoch {epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Time: {epoch_dur:.2f}s | Elapsed: {time.time()-start_time:.1f}s", flush=True)

    # Save model weights and training history
    model_save_path = MODELS_DIR / f"{genre}_lstm_numpy.npz"
    model.save_weights(str(model_save_path))

    history_save_path = MODELS_DIR / f"{genre}_history.json"
    with open(history_save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n[OK] Training Complete for {genre}!", flush=True)
    print(f"     Weights saved to: {model_save_path}", flush=True)
    print(f"     History saved to: {history_save_path}", flush=True)
    return model, history

def train_torch_lstm(genre="classical", epochs=30, batch_size=64, lr=0.002):
    """
    Trains TorchMusicLSTM using PyTorch, Adam optimizer and CrossEntropyLoss.
    """
    if not HAS_TORCH:
        print("[!] PyTorch not found. Falling back to NumPy LSTM engine.")
        return train_numpy_lstm(genre, epochs=epochs)

    data_path = DATA_DIR / f"{genre}_processed.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab_size = data["vocab_size"]
    inputs = torch.tensor(data["inputs"], dtype=torch.long)
    outputs = torch.tensor(data["outputs"], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training PyTorch LSTM on {device} (Vocab size: {vocab_size}, Samples: {len(inputs)})")

    from src.model import TorchMusicLSTM
    model = TorchMusicLSTM(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2).to(device)

    dataset = TensorDataset(inputs, outputs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        avg_loss = total_loss / len(dataset)
        history["epoch"].append(epoch)
        history["loss"].append(round(avg_loss, 4))
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Torch Epoch {epoch}/{epochs}] Loss: {avg_loss:.4f}")

    save_path = MODELS_DIR / f"{genre}_lstm_torch.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[OK] PyTorch model saved to {save_path}")
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", type=str, default="classical", choices=["classical", "jazz"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--engine", type=str, default="auto", choices=["auto", "numpy", "torch"])
    args = parser.parse_args()

    if args.engine == "torch" or (args.engine == "auto" and HAS_TORCH):
        train_torch_lstm(genre=args.genre, epochs=args.epochs)
    else:
        train_numpy_lstm(genre=args.genre, epochs=args.epochs, max_samples=args.samples)
