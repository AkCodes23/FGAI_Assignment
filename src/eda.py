"""
src/eda.py
Module: Exploratory Data Analysis (EDA) and Corpus Diagnostics for Autoregressive LM.

Context Acquisition:
- Service/Module: src/eda.py
- Responsibility: Execute corpus diagnostics, compute lexical diversity metrics, test Zipf's Law,
  compute unigram entropy, extract n-grams, and produce publication-grade analytical plots.
- Upstream Dependencies: data/sherlock_holmes.txt, src/data.py
- Downstream Dependencies: main.py, scripts/generate_submission_doc.py, scripts/build_notebook.py
- Data Ownership: Generates results/eda_*.png, results/system_architecture_diagram.png, and results/eda_*.csv.
- Runtime Constraints: < 10 seconds execution time on CPU, RAM < 300 MB.
- Failure Expectations: Assert file existence, handle division by zero safely, log errors explicitly.
"""

from collections import Counter
import math
import os
import re
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from src.data import (
    clean_gutenberg_text,
    download_corpus,
    tokenize_text,
    LOCAL_CORPUS_PATH,
    SPECIAL_TOKENS,
)


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DPI = 300
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
ACCENT_COLOR = "#2ca02c"
DARK_GRAY = "#2b2b2b"


def compute_sentence_statistics(raw_text: str) -> Dict[str, Any]:
    """
    Splits raw text into sentences and computes robust descriptive statistics.
    """
    clean_text = clean_gutenberg_text(raw_text)
    # Split across punctuation boundaries followed by whitespace or line breaks
    raw_sentences = re.split(r'(?<=[.!?])\s+', clean_text.strip())
    # Calculate word length per sentence
    sentence_lengths = [len(re.findall(r'\b[a-zA-Z]+(?:[\'\-][a-zA-Z]+)?\b', s)) for s in raw_sentences]
    sentence_lengths = [l for l in sentence_lengths if l > 0]

    arr = np.array(sentence_lengths)
    q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q75 - q25

    return {
        "total_sentences": int(len(arr)),
        "mean_length": float(np.mean(arr)),
        "std_length": float(np.std(arr)),
        "median_length": float(np.median(arr)),
        "q25_length": float(q25),
        "q75_length": float(q75),
        "iqr_length": float(iqr),
        "min_length": int(np.min(arr)),
        "max_length": int(np.max(arr)),
        "p95_length": float(np.percentile(arr, 95)),
        "p99_length": float(np.percentile(arr, 99)),
        "lengths_array": arr,
    }


def compute_lexical_and_information_metrics(tokens: List[str]) -> Dict[str, Any]:
    """
    Computes lexical diversity, Zipfian exponent, and unigram entropy.
    """
    total_tokens = len(tokens)
    if total_tokens == 0:
        raise ValueError("Token sequence cannot be empty for lexical analysis.")

    freqs = Counter(tokens)
    unique_tokens = len(freqs)

    # Type-Token Ratio & Guiraud's Index
    ttr = unique_tokens / total_tokens
    root_ttr = unique_tokens / math.sqrt(total_tokens)

    # Hapax Legomena (frequency = 1) & Dis Legomena (frequency = 2)
    hapax_count = sum(1 for c in freqs.values() if c == 1)
    dis_count = sum(1 for c in freqs.values() if c == 2)
    hapax_pct = (hapax_count / unique_tokens) * 100.0
    dis_pct = (dis_count / unique_tokens) * 100.0

    # Shannon Unigram Entropy (bits/token) and Perplexity Lower Bound
    probs = np.array(list(freqs.values()), dtype=np.float64) / total_tokens
    unigram_entropy = -np.sum(probs * np.log2(probs))
    zeroth_order_ppl = 2.0 ** unigram_entropy

    # Zipf's Law OLS Regression on top 1000 ranks
    sorted_counts = np.array(sorted(freqs.values(), reverse=True))
    ranks = np.arange(1, len(sorted_counts) + 1)
    k_fit = min(1000, len(sorted_counts))
    log_ranks = np.log10(ranks[:k_fit])
    log_freqs = np.log10(sorted_counts[:k_fit])
    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    zipf_alpha = -slope
    r2_score = float(np.corrcoef(log_ranks, log_freqs)[0, 1] ** 2)

    # Cumulative Coverage thresholds
    cum_counts = np.cumsum(sorted_counts)
    coverage_pcts = (cum_counts / total_tokens) * 100.0
    cutoff_50 = int(np.searchsorted(coverage_pcts, 50.0)) + 1
    cutoff_80 = int(np.searchsorted(coverage_pcts, 80.0)) + 1
    cutoff_90 = int(np.searchsorted(coverage_pcts, 90.0)) + 1
    cutoff_95 = int(np.searchsorted(coverage_pcts, 95.0)) + 1

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "ttr": ttr,
        "root_ttr": root_ttr,
        "hapax_count": hapax_count,
        "hapax_pct": hapax_pct,
        "dis_count": dis_count,
        "dis_pct": dis_pct,
        "unigram_entropy_bits": float(unigram_entropy),
        "zeroth_order_ppl": float(zeroth_order_ppl),
        "zipf_alpha": float(zipf_alpha),
        "zipf_intercept": float(intercept),
        "zipf_r2": float(r2_score),
        "cutoff_50_rank": cutoff_50,
        "cutoff_80_rank": cutoff_80,
        "cutoff_90_rank": cutoff_90,
        "cutoff_95_rank": cutoff_95,
        "sorted_counts": sorted_counts,
        "ranks": ranks,
        "coverage_pcts": coverage_pcts,
        "freqs": freqs,
    }


def compute_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, int, float]]:
    """
    Extracts top n-grams from token stream excluding pure punctuation tokens.
    """
    valid_tokens = [t for t in tokens if re.search(r'[a-zA-Z]', t)]
    ngrams_list = [" ".join(valid_tokens[i : i + n]) for i in range(len(valid_tokens) - n + 1)]
    ngram_counts = Counter(ngrams_list)
    total_ngrams = len(ngrams_list)

    results = []
    for ngram, count in ngram_counts.most_common(25):
        prop = (count / total_ngrams) * 100.0
        results.append((ngram, count, prop))
    return results


def plot_sentence_length_distribution(sent_stats: Dict[str, Any], output_path: str) -> None:
    """
    Generates a research-grade sentence length histogram with inset boxplot and statistics.
    """
    lengths = sent_stats["lengths_array"]
    # Filter extreme outliers (> 99th percentile) for visual clarity in histogram
    clip_val = int(sent_stats["p99_length"]) + 5
    clipped_lengths = lengths[lengths <= clip_val]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    ax.set_facecolor("#fcfcfc")
    ax.grid(color="#e0e0e0", linestyle="--", linewidth=0.7, alpha=0.7, zorder=1)

    bins = np.arange(1, clip_val + 2, 2)
    n_vals, bins_vals, patches_objs = ax.hist(
        clipped_lengths,
        bins=bins,
        color="#2b5c8f",
        edgecolor="#18385c",
        alpha=0.85,
        density=True,
        zorder=2,
        label="Sentence Length Frequency",
    )

    # Vertical reference lines
    mean_val = sent_stats["mean_length"]
    median_val = sent_stats["median_length"]
    ax.axvline(mean_val, color="#d9534f", linestyle="--", linewidth=2.0, label=f"Mean: {mean_val:.2f} words", zorder=3)
    ax.axvline(median_val, color="#f0ad4e", linestyle="-.", linewidth=2.0, label=f"Median: {median_val:.1f} words", zorder=3)
    ax.axvline(sent_stats["p95_length"], color="#5cb85c", linestyle=":", linewidth=2.0, label=f"95th Pct: {sent_stats['p95_length']:.0f} words", zorder=3)

    ax.set_title("Sentence Length Distribution: The Adventures of Sherlock Holmes", fontsize=13, fontweight="bold", pad=12, color=DARK_GRAY)
    ax.set_xlabel("Sentence Length (Words per Sentence)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5, loc="upper right")

    # Inset Summary Statistics Table
    stats_text = (
        f"Corpus Metrics\n"
        f"--------------------\n"
        f"Sentences: {sent_stats['total_sentences']:,}\n"
        f"Mean: {mean_val:.2f} ± {sent_stats['std_length']:.2f}\n"
        f"Median (Q2): {median_val:.1f}\n"
        f"IQR [Q1, Q3]: [{sent_stats['q25_length']:.1f}, {sent_stats['q75_length']:.1f}]\n"
        f"Range: [{sent_stats['min_length']}, {sent_stats['max_length']}] words"
    )
    ax.text(
        0.58, 0.45, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95),
        zorder=4,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_zipf_law_analysis(lex_stats: Dict[str, Any], output_path: str) -> None:
    """
    Plots the empirical log-rank vs log-frequency Zipfian distribution with linear regression.
    """
    ranks = lex_stats["ranks"]
    counts = lex_stats["sorted_counts"]
    alpha = lex_stats["zipf_alpha"]
    intercept = lex_stats["zipf_intercept"]
    r2 = lex_stats["zipf_r2"]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=DPI)
    ax.set_facecolor("#fcfcfc")
    ax.grid(color="#e0e0e0", linestyle="--", linewidth=0.7, alpha=0.7, zorder=1)

    # Plot empirical data
    ax.loglog(ranks, counts, marker=".", markersize=3, linestyle="none", color="#1f77b4", alpha=0.6, label="Empirical Word Frequencies", zorder=2)

    # Fitted Regression Line (top 1000 ranks)
    fit_ranks = np.logspace(0, np.log10(1000), 100)
    fit_freqs = (10 ** intercept) * (fit_ranks ** (-alpha))
    ax.loglog(fit_ranks, fit_freqs, color="#d9534f", linewidth=2.2, linestyle="-", label=rf"Power Law Fit: $\alpha={alpha:.3f}$ ($R^2={r2:.4f}$)", zorder=3)

    # Theoretical Ideal Zipf Line (alpha = 1.0)
    ideal_freqs = counts[0] * (ranks ** (-1.0))
    ax.loglog(ranks, ideal_freqs, color="#6c757d", linewidth=1.5, linestyle="--", label=r"Ideal Zipfian Model ($\alpha=1.00$)", zorder=2)

    # Annotate landmark literary tokens
    landmarks = [
        (1, counts[0], "1: 'the'"),
        (2, counts[1], "2: 'and'"),
        (10, counts[9], "10: 'it'"),
        (50, counts[49], "50: 'mr'"),
        (110, counts[109], "110: 'holmes'"),
        (450, counts[449], "450: 'watson'"),
    ]
    for r_pos, f_pos, label_txt in landmarks:
        ax.annotate(
            label_txt,
            xy=(r_pos, f_pos),
            xytext=(r_pos * 1.5, f_pos * 1.8),
            arrowprops=dict(facecolor="#2b2b2b", arrowstyle="->", lw=0.8),
            fontsize=8.5,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#adb5bd", alpha=0.9),
            zorder=4,
        )

    ax.set_title("Empirical Word Frequency Distribution & Zipf's Law Fit", fontsize=13, fontweight="bold", pad=12, color=DARK_GRAY)
    ax.set_xlabel("Token Rank $r$ (Log Scale)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Occurrence Frequency $f(r)$ (Log Scale)", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=9.5, loc="lower left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_token_coverage(lex_stats: Dict[str, Any], output_path: str, active_vocab_size: int = 4378) -> None:
    """
    Plots cumulative token coverage versus vocabulary rank with milestone markers.
    """
    ranks = lex_stats["ranks"]
    coverage = lex_stats["coverage_pcts"]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    ax.set_facecolor("#fcfcfc")
    ax.grid(color="#e0e0e0", linestyle="--", linewidth=0.7, alpha=0.7, zorder=1)

    ax.plot(ranks, coverage, color="#198754", linewidth=2.2, label="Cumulative Token Coverage (%)", zorder=2)
    ax.fill_between(ranks, 0, coverage, color="#198754", alpha=0.1, zorder=1)

    # Reference coverage lines
    milestones = [
        (lex_stats["cutoff_50_rank"], 50.0, "50% Coverage (Top 57 words)"),
        (lex_stats["cutoff_80_rank"], 80.0, "80% Coverage (Top 709 words)"),
        (lex_stats["cutoff_95_rank"], 95.0, "95% Coverage (Top 3,712 words)"),
    ]
    colors = ["#fd7e14", "#d63384", "#6f42c1"]
    for (rk, pct, label_str), col in zip(milestones, colors):
        ax.scatter([rk], [pct], color=col, s=40, zorder=4)
        ax.axhline(pct, color=col, linestyle=":", linewidth=1.0, alpha=0.7)
        ax.annotate(
            label_str,
            xy=(rk, pct),
            xytext=(rk * 1.4, pct - 5),
            fontsize=8.5,
            fontweight="semibold",
            color=col,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor=col, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
            zorder=5,
        )

    # Mark active vocabulary threshold
    active_cov = float(coverage[min(active_vocab_size - 1, len(coverage) - 1)])
    ax.axvline(active_vocab_size, color="#dc3545", linestyle="--", linewidth=1.8, label=f"Active Vocabulary Limit ($V={active_vocab_size:,}$, {active_cov:.1f}%)", zorder=3)
    ax.scatter([active_vocab_size], [active_cov], color="#dc3545", s=60, zorder=6)

    ax.set_xscale("log")
    ax.set_ylim(0, 102)
    ax.set_title("Lexical Coverage Curve vs. Vocabulary Size", fontsize=13, fontweight="bold", pad=12, color=DARK_GRAY)
    ax.set_xlabel("Vocabulary Size / Rank (Log Scale)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Cumulative Corpus Tokens Covered (%)", fontsize=11, fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=9.5, loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_top_ngrams(unigrams: List[Tuple[str, int, float]], bigrams: List[Tuple[str, int, float]], output_path: str) -> None:
    """
    Plots two-panel horizontal bar charts for Top-12 unigrams and Top-12 bigrams.
    """
    top_u = unigrams[:12][::-1]
    top_b = bigrams[:12][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)

    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfc")
        ax.grid(axis="x", color="#e0e0e0", linestyle="--", linewidth=0.7, alpha=0.7, zorder=1)

    # Panel A: Unigrams
    words_u = [w for w, _, _ in top_u]
    counts_u = [c for _, c, _ in top_u]
    bars1 = ax1.barh(words_u, counts_u, color="#2b5c8f", edgecolor="#18385c", height=0.65, zorder=2)
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 80, bar.get_y() + bar.get_height() / 2, f"{int(width):,}", va="center", ha="left", fontsize=8.5, color="#2b2b2b")

    ax1.set_title("(A) Top 12 Unigram Word Tokens", fontsize=11.5, fontweight="bold", color=DARK_GRAY, pad=10)
    ax1.set_xlabel("Occurrence Frequency", fontsize=10, fontweight="semibold")
    ax1.set_xlim(0, max(counts_u) * 1.15)

    # Panel B: Bigrams
    words_b = [b for b, _, _ in top_b]
    counts_b = [c for _, c, _ in top_b]
    bars2 = ax2.barh(words_b, counts_b, color="#d9534f", edgecolor="#a93226", height=0.65, zorder=2)
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 20, bar.get_y() + bar.get_height() / 2, f"{int(width):,}", va="center", ha="left", fontsize=8.5, color="#2b2b2b")

    ax2.set_title("(B) Top 12 Frequent Bigram Collocations", fontsize=11.5, fontweight="bold", color=DARK_GRAY, pad=10)
    ax2.set_xlabel("Occurrence Frequency", fontsize=10, fontweight="semibold")
    ax2.set_xlim(0, max(counts_b) * 1.18)

    plt.suptitle("Lexical N-Gram Frequency Profiles: Sherlock Holmes Corpus", fontsize=13, fontweight="bold", y=0.98, color=DARK_GRAY)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_system_architecture_diagram(output_path: str) -> None:
    """
    Renders a publication-grade vector architectural flowchart illustrating the end-to-end
    autoregressive language modeling and sequential generation system.
    """
    fig, ax = plt.subplots(figsize=(14, 8), dpi=DPI)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.8,rounding_size=2.0",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.8,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1c2833", zorder=3)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=8.2, color="#4a5568", zorder=3)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(facecolor="#2b2b2b", edgecolor="#2b2b2b", width=1.5, headwidth=7, headlength=7, shrink=0.05),
            zorder=4,
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 1.8, label, ha="center", va="bottom", fontsize=7.8, fontweight="semibold", color="#2c3e50", zorder=5)

    # Title Banner
    ax.text(50, 96, "Autoregressive Text Generation System Architecture", ha="center", va="center", fontsize=15, fontweight="bold", color="#1a252f")
    ax.text(50, 92, "Deep Sequential Recurrent Neural Framework (Simple RNN, BiRNN, LSTM, GRU)", ha="center", va="center", fontsize=10, color="#566573")

    # Stage 1: Ingestion & Preprocessing
    draw_box(4, 68, 22, 16, "1. Corpus Ingestion", "Sherlock Holmes (~108k tokens)\nGutenberg Clean & Lowercase", "#ebf5fb", "#2980b9")
    draw_arrow(26, 76, 36, 76, "Text stream")

    # Stage 2: Tokenization & Sequence Construction
    draw_box(36, 68, 26, 16, "2. Causal Sliding Window", "Context Length: L = 20\nStride: s = 2, Vocab V = 4,378", "#eafaf1", "#27ae60")
    draw_arrow(62, 76, 72, 76, "Tensors (B=64, L=20)")

    # Stage 3: Embedding Layer
    draw_box(72, 68, 24, 16, "3. Embedding Layer", "Lookup Matrix: V × d_emb\nDim: 4,378 × 128", "#fef9e7", "#f39c12")
    draw_arrow(84, 68, 84, 52, "Embedded Vectors\n(B, 20, 128)")

    # Stage 4: Recurrent Model Zoo (Central Hub)
    rect_zoo = patches.FancyBboxPatch(
        (10, 22), 80, 30,
        boxstyle="round,pad=1.0,rounding_size=2.5",
        facecolor="#f8f9fa",
        edgecolor="#6c757d",
        linestyle="--",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(rect_zoo)
    ax.text(50, 48.5, "4. Comparable Recurrent Latent Encoders (H = 256, Layers = 2, Dropout = 0.2)", ha="center", va="center", fontsize=11, fontweight="bold", color="#2c3e50")

    draw_box(13, 26, 17, 18, "Simple RNN", "1.91M Parameters\nVanilla Jordan/Elman", "#fdedec", "#e74c3c")
    draw_box(32, 26, 17, 18, "Bidirectional RNN", "1.85M Parameters\nCausal Inference Mode", "#f5eef8", "#8e44ad")
    draw_box(51, 26, 17, 18, "LSTM", "2.61M Parameters\nInput, Forget, Output Gates", "#e8f8f5", "#16a085")
    draw_box(70, 26, 17, 18, "GRU (Champion)", "2.38M Parameters\nReset & Update Gates", "#d4efdf", "#229954")

    # Stage 5: Output Projection & Sampling
    draw_arrow(50, 22, 50, 14, "Last Hidden State h_T (B, 256)")
    draw_box(20, 2, 60, 12, "5. Linear Projection & Controlled Stochastic Sampling", "Dense Layer (256 → 4,378 Logits)  |  Mask Special Tokens (<pad>, <unk>)  |  Temperature T ∈ {0.2, 0.7, 1.2}", "#ebedef", "#34495e")

    # Feedback Loop: Autoregressive Step
    ax.annotate(
        "Append Predicted Token x_{t+1} & Slide Window",
        xy=(4, 76),
        xytext=(4, 8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4", color="#7f8c8d", lw=1.8, linestyle=":"),
        ha="left", va="center", fontsize=8.5, fontweight="semibold", color="#7f8c8d", zorder=5,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_full_eda() -> Dict[str, Any]:
    """
    Orchestrates the end-to-end EDA pipeline: statistics computation, tabular export,
    and publication-grade visualization rendering.
    """
    print("\n" + "="*70)
    print("RUNNING CORPUS EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)

    raw_text = download_corpus()
    clean_text = clean_gutenberg_text(raw_text)
    tokens = tokenize_text(clean_text)

    # 1. Sentence Statistics
    sent_stats = compute_sentence_statistics(raw_text)
    print(f"[EDA] Total Sentences: {sent_stats['total_sentences']:,}")
    print(f"[EDA] Sentence Length: {sent_stats['mean_length']:.2f} ± {sent_stats['std_length']:.2f} words (Median: {sent_stats['median_length']:.1f})")

    # 2. Lexical and Information Metrics
    lex_stats = compute_lexical_and_information_metrics(tokens)
    print(f"[EDA] Total Tokens: {lex_stats['total_tokens']:,} | Unique: {lex_stats['unique_tokens']:,}")
    print(f"[EDA] Type-Token Ratio (TTR): {lex_stats['ttr']:.4f} | Guiraud Index: {lex_stats['root_ttr']:.2f}")
    print(f"[EDA] Hapax Legomena: {lex_stats['hapax_count']:,} ({lex_stats['hapax_pct']:.2f}%)")
    print(f"[EDA] Dis Legomena: {lex_stats['dis_count']:,} ({lex_stats['dis_pct']:.2f}%)")
    print(f"[EDA] Unigram Entropy: {lex_stats['unigram_entropy_bits']:.4f} bits/token")
    print(f"[EDA] Theoretical Zeroth-Order Perplexity: {lex_stats['zeroth_order_ppl']:.2f}")
    print(f"[EDA] Zipf Exponent (alpha): {lex_stats['zipf_alpha']:.4f} (R^2 = {lex_stats['zipf_r2']:.4f})")

    # 3. N-grams
    unigrams = compute_ngrams(tokens, n=1)
    bigrams = compute_ngrams(tokens, n=2)

    # 4. Generate Figures
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sent_path = os.path.join(RESULTS_DIR, "eda_sentence_lengths.png")
    zipf_path = os.path.join(RESULTS_DIR, "eda_zipf_law.png")
    cov_path = os.path.join(RESULTS_DIR, "eda_token_coverage.png")
    ngram_path = os.path.join(RESULTS_DIR, "eda_top_ngrams.png")
    arch_path = os.path.join(RESULTS_DIR, "system_architecture_diagram.png")

    plot_sentence_length_distribution(sent_stats, sent_path)
    plot_zipf_law_analysis(lex_stats, zipf_path)
    plot_token_coverage(lex_stats, cov_path, active_vocab_size=4378)
    plot_top_ngrams(unigrams, bigrams, ngram_path)
    plot_system_architecture_diagram(arch_path)

    # 5. Export Summary CSVs
    corpus_summary_csv = os.path.join(RESULTS_DIR, "eda_corpus_stats.csv")
    summary_df = pd.DataFrame([
        {"Metric": "Raw Characters", "Value": f"{len(raw_text):,}"},
        {"Metric": "Clean Characters", "Value": f"{len(clean_text):,}"},
        {"Metric": "Total Corpus Tokens", "Value": f"{lex_stats['total_tokens']:,}"},
        {"Metric": "Unique Tokens (Raw Vocabulary)", "Value": f"{lex_stats['unique_tokens']:,}"},
        {"Metric": "Effective Model Vocabulary Size", "Value": "4,378"},
        {"Metric": "Type-Token Ratio (TTR)", "Value": f"{lex_stats['ttr']:.4f}"},
        {"Metric": "Guiraud's Index (Root TTR)", "Value": f"{lex_stats['root_ttr']:.2f}"},
        {"Metric": "Hapax Legomena Count (freq = 1)", "Value": f"{lex_stats['hapax_count']:,} ({lex_stats['hapax_pct']:.2f}%)"},
        {"Metric": "Dis Legomena Count (freq = 2)", "Value": f"{lex_stats['dis_count']:,} ({lex_stats['dis_pct']:.2f}%)"},
        {"Metric": "Unigram Shannon Entropy", "Value": f"{lex_stats['unigram_entropy_bits']:.4f} bits/token"},
        {"Metric": "Theoretical Zeroth-Order Perplexity", "Value": f"{lex_stats['zeroth_order_ppl']:.2f}"},
        {"Metric": "Empirical Zipf Exponent (alpha)", "Value": f"{lex_stats['zipf_alpha']:.4f}"},
        {"Metric": "Zipf Power Law Fit R^2", "Value": f"{lex_stats['zipf_r2']:.4f}"},
        {"Metric": "Top 50% Token Coverage Threshold", "Value": f"Top {lex_stats['cutoff_50_rank']} words"},
        {"Metric": "Top 80% Token Coverage Threshold", "Value": f"Top {lex_stats['cutoff_80_rank']} words"},
        {"Metric": "Top 95% Token Coverage Threshold", "Value": f"Top {lex_stats['cutoff_95_rank']} words"},
        {"Metric": "Total Sentences Extracted", "Value": f"{sent_stats['total_sentences']:,}"},
        {"Metric": "Mean Sentence Length", "Value": f"{sent_stats['mean_length']:.2f} ± {sent_stats['std_length']:.2f} words"},
        {"Metric": "Median Sentence Length", "Value": f"{sent_stats['median_length']:.1f} words"},
        {"Metric": "IQR Sentence Length", "Value": f"[{sent_stats['q25_length']:.1f}, {sent_stats['q75_length']:.1f}] words"},
        {"Metric": "95th Percentile Sentence Length", "Value": f"{sent_stats['p95_length']:.1f} words"},
    ])
    summary_df.to_csv(corpus_summary_csv, index=False)
    print(f"Exported: {corpus_summary_csv}")

    ngram_csv = os.path.join(RESULTS_DIR, "eda_top_ngrams.csv")
    ngram_rows = []
    for rank, (u, c, p) in enumerate(unigrams[:25], 1):
        b, bc, bp = bigrams[rank - 1] if rank - 1 < len(bigrams) else ("", 0, 0.0)
        ngram_rows.append({
            "Rank": rank,
            "Unigram": u,
            "Unigram_Count": c,
            "Unigram_Pct": f"{p:.3f}%",
            "Bigram": b,
            "Bigram_Count": bc,
            "Bigram_Pct": f"{bp:.3f}%",
        })
    pd.DataFrame(ngram_rows).to_csv(ngram_csv, index=False)
    print(f"Exported: {ngram_csv}")

    print("="*70)
    print("EDA EXECUTION COMPLETE: All figures and tables generated.")
    print("="*70 + "\n")

    return {
        "sent_stats": sent_stats,
        "lex_stats": lex_stats,
        "unigrams": unigrams,
        "bigrams": bigrams,
    }


if __name__ == "__main__":
    run_full_eda()
