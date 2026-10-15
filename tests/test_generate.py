"""
tests/test_generate.py
Unit tests for sampling mechanisms, text generation, and qualitative diversity metrics.
"""

import pytest
import torch
from src.data import Vocabulary
from src.generate import (
    compute_diversity_metrics,
    generate_text,
    sample_next_token,
)
from src.models import GRULM


def test_sample_next_token_greedy():
    # Large logit at index 5
    logits = torch.tensor([-10.0, -5.0, 0.0, 1.0, 2.0, 100.0, -2.0])
    # Near zero temperature should pick greedy max index 5
    sampled_idx = sample_next_token(logits, temperature=0.01)
    assert sampled_idx == 5


def test_compute_diversity_metrics():
    text_diverse = "holmes examined the mysterious silver watch carefully under bright lamp"
    d1, d2, rep = compute_diversity_metrics(text_diverse)

    assert 0.0 <= d1 <= 1.0
    assert 0.0 <= d2 <= 1.0
    assert rep == 0.0  # No repeating trigrams

    text_repetitive = "the dog ran the dog ran the dog ran the dog ran"
    d1_rep, d2_rep, rep_rate = compute_diversity_metrics(text_repetitive)
    assert rep_rate > 0.0
    assert d1_rep < d1


def test_autoregressive_generate_text():
    tokens = ["the", "curious", "incident", "of", "the", "dog", "in", "the", "night", "time"]
    vocab = Vocabulary(min_freq=1, max_size=50)
    vocab.build_vocab(tokens)

    model = GRULM(vocab_size=len(vocab), embed_dim=16, hidden_dim=32, num_layers=1)
    prompt = "the curious incident"
    output = generate_text(model, vocab, prompt=prompt, max_new_tokens=10, temperature=0.7)

    assert isinstance(output, str)
    assert len(output) > len(prompt)
    assert "the curious incident" in output
