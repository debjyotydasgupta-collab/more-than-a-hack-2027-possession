"""
src/models/sequential_nn.py
===========================
PyTorch Bi-LSTM / TCN sequence architecture for spatial-temporal sequence learning.
Gracefully skipped if torch is not installed.
"""

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    pass

if HAS_TORCH:
    class SequentialPossessionNet(nn.Module):
        """2-layer Dilated Temporal Convolutional Network + Bi-LSTM."""

        def __init__(self, in_features=92, hidden_dim=128, num_classes=4, dropout=0.3):
            super().__init__()
            self.tcn1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1, dilation=1)
            self.tcn2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
            self.act = nn.Mish()
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout
            )

            self.fc = nn.Linear(hidden_dim * 2, num_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # Support both (Batch, Features) and (Batch, Seq_len, Features)
            is_2d = (x.dim() == 2)
            if is_2d:
                x = x.unsqueeze(1)

            # Permute for 1D conv: (Batch, Features, Seq_len)
            x_in = x.transpose(1, 2)
            h = self.norm1(self.act(self.tcn1(x_in)))
            h = self.norm2(self.act(self.tcn2(h)))

            # Back to (Batch, Seq_len, Hidden)
            h = h.transpose(1, 2)
            lstm_out, _ = self.lstm(h)
            lstm_out = self.dropout(lstm_out)

            # Logits: (Batch, Seq_len, num_classes)
            logits = self.fc(lstm_out)
            if is_2d:
                logits = logits.squeeze(1)
            return logits

        def predict_proba(self, x):
            """Return (N, 4) probability distribution over [H, A, D, X]."""
            self.eval()
            with torch.no_grad():
                if not isinstance(x, torch.Tensor):
                    x_t = torch.tensor(x, dtype=torch.float32)
                else:
                    x_t = x.float()
                logits = self.forward(x_t)
                probs = torch.softmax(logits, dim=-1)
                return probs.cpu().numpy()
else:
    class SequentialPossessionNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SequentialPossessionNet.")

