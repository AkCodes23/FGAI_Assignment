"""
tests/test_data.py
Unit tests for data cleaning, tokenization, vocabulary, and sliding-window dataset creation.
"""

import pytest
import torch
from src.data import (
    AutoregressiveTextDataset,
    Vocabulary,
    clean_gutenberg_text,
    tokenize_text,
    PAD_IDX,
    UNK_IDX,
)


def test_clean_gutenberg_text():
    sample = (
        "Random Header\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SHERLOCK HOLMES ***\n"
        "To Sherlock Holmes she is always THE woman.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SHERLOCK HOLMES ***\n"
        "Footer disclaimer"
    )
    cleaned = clean_gutenberg_text(sample)
    assert "Random Header" not in cleaned
    assert "Footer disclaimer" not in cleaned
    assert "To Sherlock Holmes she is always THE woman." in cleaned


def test_tokenize_text():
    sample = "Holmes said, 'It's a elementary problem, Watson!'"
    tokens = tokenize_text(sample)
    expected_subset = ["holmes", "said", ",", "'", "it's", "a", "elementary", "problem", ",", "watson", "!", "'"]
    # Check that words and punctuation are captured
    assert "holmes" in tokens
    assert "it's" in tokens
    assert "," in tokens
    assert "!" in tokens
    assert all(t.islower() or not t.isalpha() for t in tokens)


def test_vocabulary_build_and_lookup():
    tokens = ["the", "dog", "barked", "at", "the", "cat", "the", "cat", "ran"]
    vocab = Vocabulary(min_freq=1, max_size=20)
    vocab.build_vocab(tokens)

    assert len(vocab) > 4  # Special tokens + unique tokens
    assert vocab.lookup_index("<pad>") == PAD_IDX
    assert vocab.lookup_index("<unk>") == UNK_IDX
    assert vocab.lookup_index("the") > 3
    assert vocab.lookup_index("nonexistent_token_xyz") == UNK_IDX

    # Decode roundtrip
    indices = vocab.numericalize(["the", "cat", "ran"])
    decoded = vocab.decode(indices)
    assert "the cat ran" in decoded


def test_autoregressive_dataset_shapes_and_causality():
    token_ids = list(range(100))  # Synthetic sequential tokens: 0, 1, 2, ..., 99
    seq_len = 10
    step_size = 1

    dataset = AutoregressiveTextDataset(token_ids, seq_len=seq_len, step_size=step_size)
    assert len(dataset) == 100 - seq_len

    # First pair: input [0..9], target 10
    inp, tgt = dataset[0]
    assert inp.shape == torch.Size([seq_len])
    assert tgt.item() == 10
    assert torch.equal(inp, torch.tensor(list(range(10)), dtype=torch.long))

    # Causality test: Target is never inside the input window
    assert tgt.item() not in inp.tolist()
