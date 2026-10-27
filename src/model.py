"""
Deep Learning Model Architecture for Music Generation
Includes:
  1. TorchMusicLSTM: Multi-layer LSTM neural network with Dropout & Embedding in PyTorch.
  2. NumpyMusicLSTM: Pure vectorized multi-gate LSTM cell with Adam optimizer.
"""

import numpy as np

# Optional PyTorch integration
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = object

if HAS_TORCH:
    class TorchMusicLSTM(nn.Module):
        """
        Deep Multi-Layer LSTM Architecture for Musical Note & Chord Prediction.
        """
        def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
            super().__init__()
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim // 2, vocab_size)

        def forward(self, x, hidden=None):
            # x: [batch_size, seq_len]
            embeds = self.embedding(x) # [batch_size, seq_len, embedding_dim]
            lstm_out, hidden = self.lstm(embeds, hidden) # [batch_size, seq_len, hidden_dim]
            # Take last timestep
            last_out = lstm_out[:, -1, :]
            out = self.dropout(last_out)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out) # [batch_size, vocab_size]
            return out, hidden
else:
    class TorchMusicLSTM:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed in the current environment.")


class NumpyMusicLSTM:
    """
    Vectorized Long Short-Term Memory (LSTM) Neural Network implemented in pure NumPy.
    Architecture:
      Input Token Indices -> Embedding Matrix -> LSTM Layer (Gates: f, i, c_tilde, o) -> Dense Softmax
    """
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, learning_rate=0.005):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Initialize Weights (Xavier/He initialization)
        concat_dim = embedding_dim + hidden_dim
        scale = np.sqrt(2.0 / (concat_dim + hidden_dim))

        # Weight matrices for 4 gates: [W_f, W_i, W_c, W_o]
        self.W = np.random.randn(4 * hidden_dim, concat_dim) * scale
        self.b = np.zeros((4 * hidden_dim, 1))

        # Forget gate bias trick (initialize to 1.0 to prevent early vanishing gradients)
        self.b[:hidden_dim, :] = 1.0

        # Embedding weights: [vocab_size, embedding_dim]
        self.W_embed = np.random.randn(vocab_size, embedding_dim) * 0.1

        # Output dense layer: [vocab_size, hidden_dim]
        self.W_out = np.random.randn(vocab_size, hidden_dim) * np.sqrt(2.0 / (hidden_dim + vocab_size))
        self.b_out = np.zeros((vocab_size, 1))

        # Adam optimizer caches
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.mW_out = np.zeros_like(self.W_out); self.vW_out = np.zeros_like(self.W_out)
        self.mb_out = np.zeros_like(self.b_out); self.vb_out = np.zeros_like(self.b_out)
        self.mW_embed = np.zeros_like(self.W_embed); self.vW_embed = np.zeros_like(self.W_embed)
        self.t = 0

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward_sequence(self, token_indices):
        """
        Processes a sequence of token IDs through the LSTM.
        Returns:
          last_hidden (h_T), final output probabilities (probs), cache for backprop
        """
        seq_len = len(token_indices)
        H = self.hidden_dim

        h = np.zeros((H, 1))
        c = np.zeros((H, 1))

        caches = []
        for idx in token_indices:
            x_emb = self.W_embed[idx : idx + 1, :].T # [embedding_dim, 1]
            concat = np.vstack((x_emb, h)) # [embedding_dim + hidden_dim, 1]

            gates = np.dot(self.W, concat) + self.b
            f = self._sigmoid(gates[0*H : 1*H, :])
            i = self._sigmoid(gates[1*H : 2*H, :])
            c_tilde = np.tanh(gates[2*H : 3*H, :])
            o = self._sigmoid(gates[3*H : 4*H, :])

            c_next = f * c + i * c_tilde
            h_next = o * np.tanh(c_next)

            caches.append((concat, c, c_next, f, i, c_tilde, o, x_emb, idx))
            h, c = h_next, c_next

        # Dense output layer
        logits = np.dot(self.W_out, h) + self.b_out
        probs = self._softmax(logits)
        return probs, h, c, caches

    def predict(self, token_indices, temperature=1.0):
        """
        Generates probability distribution with temperature scaling.
        """
        probs, h, c, _ = self.forward_sequence(token_indices)
        logits = np.dot(self.W_out, h) + self.b_out
        scaled_logits = logits / max(temperature, 0.1)
        scaled_probs = self._softmax(scaled_logits).flatten()
        return scaled_probs

    def train_step(self, token_indices, target_token_idx, clip_norm=5.0):
        """
        Backpropagation Through Time (BPTT) and Adam update for a single training sample.
        """
        probs, last_h, last_c, caches = self.forward_sequence(token_indices)
        H = self.hidden_dim

        # Cross Entropy Loss
        target_prob = max(probs[target_token_idx, 0], 1e-12)
        loss = -np.log(target_prob)

        # Output layer gradient
        dlogits = probs.copy()
        dlogits[target_token_idx, 0] -= 1.0 # [vocab_size, 1]

        dW_out = np.dot(dlogits, last_h.T)
        db_out = dlogits
        dh = np.dot(self.W_out.T, dlogits)
        dc = np.zeros((H, 1))

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dW_embed = np.zeros_like(self.W_embed)

        # Backpropagation through time
        for step in reversed(range(len(caches))):
            concat, c_prev, c_curr, f, i, c_tilde, o, x_emb, idx = caches[step]

            # Gradient into c_curr
            tanh_c = np.tanh(c_curr)
            do = dh * tanh_c
            dc = dc + (dh * o * (1 - tanh_c**2))

            # Gate gradients
            df = dc * c_prev * (f * (1 - f))
            di = dc * c_tilde * (i * (1 - i))
            dc_tilde = dc * i * (1 - c_tilde**2)
            do_act = do * (o * (1 - o))

            dgates = np.vstack((df, di, dc_tilde, do_act)) # [4H, 1]
            dW += np.dot(dgates, concat.T)
            db += dgates

            dconcat = np.dot(self.W.T, dgates)
            dx_emb = dconcat[:self.embedding_dim, :]
            dh = dconcat[self.embedding_dim:, :]
            dc = dc * f

            dW_embed[idx] += dx_emb.flatten()

        # Gradient clipping
        for grad in [dW, db, dW_out, db_out, dW_embed]:
            np.clip(grad, -clip_norm, clip_norm, out=grad)

        # Adam optimizer update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for param, grad, m, v in [
            (self.W, dW, self.mW, self.vW),
            (self.b, db, self.mb, self.vb),
            (self.W_out, dW_out, self.mW_out, self.vW_out),
            (self.b_out, db_out, self.mb_out, self.vb_out),
            (self.W_embed, dW_embed, self.mW_embed, self.vW_embed),
        ]:
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        return loss

    def save_weights(self, filepath):
        """
        Saves trained model parameters to npz archive.
        """
        np.savez_compressed(
            filepath,
            W=self.W,
            b=self.b,
            W_embed=self.W_embed,
            W_out=self.W_out,
            b_out=self.b_out,
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        print(f"[OK] Model weights saved to {filepath}")

    @classmethod
    def load_weights(cls, filepath):
        """
        Loads trained model from npz archive.
        """
        data = np.load(filepath)
        model = cls(
            vocab_size=int(data["vocab_size"]),
            embedding_dim=int(data["embedding_dim"]),
            hidden_dim=int(data["hidden_dim"])
        )
        model.W = data["W"]
        model.b = data["b"]
        model.W_embed = data["W_embed"]
        model.W_out = data["W_out"]
        model.b_out = data["b_out"]
        print(f"[OK] Model weights loaded from {filepath}")
        return model
