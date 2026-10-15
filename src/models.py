"""
src/models.py
Module: Recurrent Neural Network Architectures for Autoregressive Text Generation.

Context Acquisition:
- Responsibility: PyTorch implementation of Simple RNN, Bidirectional RNN, LSTM, and GRU models.
- Upstream: PyTorch nn.Module, torch.Tensor.
- Downstream: src/train.py, src/evaluate.py, src/generate.py.
- Data Ownership: Model parameter weights and state dicts.
- Runtime Constraints: Forward pass < 15ms per batch on CPU; parameters ~1M-2.5M.
- Failure Expectations: Assert tensor dimensions, fail loudly on invalid sequence rank.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class BaseRecurrentLM(nn.Module):
    """
    Base class providing common embedding, dropout, and head projection,
    along with parameter counting and introspection.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_params": total, "trainable_params": trainable}


class SimpleRNNLM(BaseRecurrentLM):
    """
    Standard Elman RNN Language Model.
    h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(vocab_size, embed_dim, hidden_dim, dropout)
        self.num_layers = num_layers
        self.model_type = "Simple RNN"

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)
            hidden: Optional initial hidden state
        Returns:
            logits: (batch_size, vocab_size)
            hidden: (num_layers, batch_size, hidden_dim)
        """
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.rnn(embedded, hidden)
        last_step = self.dropout(out[:, -1, :])
        logits = self.fc_out(last_step)
        return logits, hidden


class BiRNNLM(BaseRecurrentLM):
    """
    Bidirectional RNN Language Model.
    Processes context window x_{1:L} forward (1 -> L) and backward (L -> 1).
    To maintain comparable hidden dimensionality, forward and backward
    directions each use hidden_dim // 2, yielding a concatenated hidden_dim.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(vocab_size, embed_dim, hidden_dim, dropout)
        self.num_layers = num_layers
        self.model_type = "Bidirectional RNN"
        self.half_hidden = hidden_dim // 2

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=self.half_hidden,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.rnn(embedded, hidden)
        # out shape: (batch_size, seq_len, 2 * half_hidden) == (batch_size, seq_len, hidden_dim)
        last_step = self.dropout(out[:, -1, :])
        logits = self.fc_out(last_step)
        return logits, hidden


class LSTMLM(BaseRecurrentLM):
    """
    Long Short-Term Memory (LSTM) Language Model.
    Utilizes input, forget, output gates and memory cell state C_t.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(vocab_size, embed_dim, hidden_dim, dropout)
        self.num_layers = num_layers
        self.model_type = "LSTM"

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.dropout(self.embedding(x))
        out, state = self.lstm(embedded, state)
        last_step = self.dropout(out[:, -1, :])
        logits = self.fc_out(last_step)
        return logits, state


class GRULM(BaseRecurrentLM):
    """
    Gated Recurrent Unit (GRU) Language Model.
    Utilizes reset gate r_t and update gate z_t.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(vocab_size, embed_dim, hidden_dim, dropout)
        self.num_layers = num_layers
        self.model_type = "GRU"

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.gru(embedded, hidden)
        last_step = self.dropout(out[:, -1, :])
        logits = self.fc_out(last_step)
        return logits, hidden


def get_model(
    model_name: str,
    vocab_size: int,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory function to instantiate models by string key.
    Allowed keys: 'rnn', 'birnn', 'lstm', 'gru'
    """
    name = model_name.lower().replace(" ", "").replace("_", "").replace("-", "")
    if name in ("rnn", "simplernn"):
        return SimpleRNNLM(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    elif name in ("birnn", "bidirectionalrnn"):
        return BiRNNLM(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    elif name == "lstm":
        return LSTMLM(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    elif name == "gru":
        return GRULM(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from 'rnn', 'birnn', 'lstm', 'gru'.")
