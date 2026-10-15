"""
tests/test_eda.py
Unit tests for EDA statistics, Zipf's law fitting, information entropy, and ngram extraction.
"""

import os
import pytest
import numpy as np

from src.eda import (
    compute_sentence_statistics,
    compute_lexical_and_information_metrics,
    compute_ngrams,
    run_full_eda,
    RESULTS_DIR,
)


def test_compute_sentence_statistics():
    text = (
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "Sherlock Holmes walked into the room. He looked around carefully! "
        "Was anyone there? Nobody answered.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***"
    )
    stats = compute_sentence_statistics(text)
    assert stats["total_sentences"] == 4
    assert stats["min_length"] >= 2
    assert stats["mean_length"] > 0
    assert stats["median_length"] > 0
    assert "lengths_array" in stats
    assert len(stats["lengths_array"]) == 4


def test_compute_lexical_and_information_metrics():
    tokens = ["the", "cat", "sat", "on", "the", "mat", "and", "the", "cat", "smiled"]
    metrics = compute_lexical_and_information_metrics(tokens)

    assert metrics["total_tokens"] == 10
    assert metrics["unique_tokens"] == 7
    assert 0 < metrics["ttr"] <= 1.0
    assert metrics["root_ttr"] > 0
    assert metrics["hapax_count"] == 5  # sat, on, mat, and, smiled
    assert metrics["dis_count"] == 1    # cat (freq 2)
    assert metrics["unigram_entropy_bits"] > 0
    assert metrics["zeroth_order_ppl"] >= 1.0
    assert metrics["zipf_alpha"] > 0
    assert 0 <= metrics["zipf_r2"] <= 1.0


def test_compute_lexical_empty_tokens_fails():
    with pytest.raises(ValueError, match="Token sequence cannot be empty"):
        compute_lexical_and_information_metrics([])


def test_compute_ngrams():
    tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    unigrams = compute_ngrams(tokens, n=1)
    bigrams = compute_ngrams(tokens, n=2)

    assert len(unigrams) > 0
    assert len(bigrams) > 0
    # First bigram is 'the quick'
    assert bigrams[0][0] == "the quick"
    assert bigrams[0][1] == 1


def test_eda_artifacts_exist():
    # Verify the generated plots and CSV tables exist and have non-zero size
    expected_files = [
        "eda_sentence_lengths.png",
        "eda_zipf_law.png",
        "eda_token_coverage.png",
        "eda_top_ngrams.png",
        "system_architecture_diagram.png",
        "eda_corpus_stats.csv",
        "eda_top_ngrams.csv",
    ]
    for fname in expected_files:
        fpath = os.path.join(RESULTS_DIR, fname)
        assert os.path.exists(fpath), f"Missing artifact: {fname}"
        assert os.path.getsize(fpath) > 500, f"Artifact {fname} is too small / empty"
