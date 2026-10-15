"""
tests/test_models.py
Unit tests for Simple RNN, Bidirectional RNN, LSTM, and GRU model architectures.
"""

import pytest
import torch
from src.models import (
    BiRNNLM,
    GRULM,
    LSTMLM,
    SimpleRNNLM,
    get_model,
)


@pytest.mark.parametrize("model_class", [SimpleRNNLM, BiRNNLM, LSTMLM, GRULM])
def test_model_forward_pass_shapes(model_class):
    vocab_size = 500
    embed_dim = 64
    hidden_dim = 128
    num_layers = 2
    batch_size = 4
    seq_len = 15

    model = model_class(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, state = model(dummy_input)

    # Output logits must match (batch_size, vocab_size)
    assert logits.shape == torch.Size([batch_size, vocab_size])
    assert not torch.isnan(logits).any()


def test_model_factory():
    v = 100
    m1 = get_model("rnn", vocab_size=v)
    m2 = get_model("birnn", vocab_size=v)
    m3 = get_model("lstm", vocab_size=v)
    m4 = get_model("gru", vocab_size=v)

    assert isinstance(m1, SimpleRNNLM)
    assert isinstance(m2, BiRNNLM)
    assert isinstance(m3, LSTMLM)
    assert isinstance(m4, GRULM)

    with pytest.raises(ValueError):
        _ = get_model("transformer_invalid", vocab_size=v)


def test_parameter_counting_and_backward():
    vocab_size = 300
    model = LSTMLM(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_layers=1)
    params = model.count_parameters()

    assert params["total_params"] > 0
    assert params["trainable_params"] == params["total_params"]

    # Verify backpropagation produces gradients for all parameters
    dummy_input = torch.randint(0, vocab_size, (2, 8))
    dummy_target = torch.randint(0, vocab_size, (2,))
    criterion = torch.nn.CrossEntropyLoss()

    logits, _ = model(dummy_input)
    loss = criterion(logits, dummy_target)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient."
