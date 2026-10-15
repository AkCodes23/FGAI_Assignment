"""
src/generate.py
Module: Autoregressive Text Generation, Temperature/Top-k/Top-p Sampling, and Qualitative Analysis.

Context Acquisition:
- Responsibility: Autoregressive sequence decoding, temperature-controlled stochastic sampling,
  and computation of qualitative metrics (Distinct-1, Distinct-2, Repetition Rate).
- Upstream: PyTorch nn.Module, src/models.py, src/data.py (Vocabulary).
- Downstream: main.py, notebooks, results/generated_samples.csv, README.md.
- Data Ownership: Generated text sequences, qualitative evaluation DataFrames.
- Runtime Constraints: < 20ms per generated token on CPU.
- Failure Expectations: Guard against empty prompts, clamp temperature to T > 0, handle unk tokens cleanly.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn
from src.data import UNK_IDX, Vocabulary

DEFAULT_SEEDS = [
    "it was a dark and stormy night when",
    "holmes sat in silence for a few minutes and then said",
    "the police had discovered no clue regarding the mystery of",
]


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> int:
    """
    Samples next token index from logits using temperature, top-k, and nucleus (top-p) sampling.
    """
    if temperature <= 0.05:
        # Near-zero temperature equates to greedy decoding
        return int(torch.argmax(logits, dim=-1).item())

    # Apply temperature
    logits = logits / max(temperature, 1e-5)

    # Optional Top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    # Optional Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def generate_text(
    model: nn.Module,
    vocab: Vocabulary,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
    seq_len: int = 25,
) -> str:
    """
    Generates text autoregressively starting from a seed prompt.
    Preserves original prompt verbatim, masks special tokens (<PAD>, <UNK>),
    and decodes fluent continuation tokens.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    prompt_trimmed = prompt.strip()
    words = prompt_trimmed.lower().split()
    if len(words) == 0:
        raise ValueError("Seed prompt cannot be empty.")

    # Convert prompt to indices
    token_ids = [vocab.lookup_index(w) for w in words]
    new_token_ids: List[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = (token_ids + new_token_ids)[-seq_len:]
            inp_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)

            logits, _ = model(inp_tensor)
            logits = logits[0].clone()  # shape: (vocab_size,)

            # Mask special tokens (<PAD>=0, <UNK>=1) so generation produces valid vocabulary
            logits[0] = -float("Inf")
            logits[1] = -float("Inf")

            # Apply repetition penalty to recently generated tokens
            if repetition_penalty > 1.0 and (token_ids + new_token_ids):
                recent_tokens = set((token_ids + new_token_ids)[-20:])
                for t in recent_tokens:
                    if logits[t] > 0:
                        logits[t] /= repetition_penalty
                    else:
                        logits[t] *= repetition_penalty

            next_idx = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            new_token_ids.append(next_idx)

    continuation = vocab.decode(new_token_ids)
    if continuation:
        return f"{prompt_trimmed} {continuation}"
    return prompt_trimmed


def compute_diversity_metrics(text: str) -> Tuple[float, float, float]:
    """
    Computes Distinct-1, Distinct-2, and 3-gram repetition rate (%) for a generated text sample.
    """
    words = [w.lower() for w in text.split() if w.isalnum()]
    if len(words) == 0:
        return 0.0, 0.0, 0.0

    # Distinct-1
    distinct_1 = len(set(words)) / len(words)

    # Distinct-2
    bigrams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
    distinct_2 = (len(set(bigrams)) / len(bigrams)) if len(bigrams) > 0 else 0.0

    # 3-gram Repetition Rate
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if len(trigrams) == 0:
        rep_rate = 0.0
    else:
        tri_counts = Counter(trigrams)
        repeated_trigrams = sum(c for c in tri_counts.values() if c > 1)
        rep_rate = (repeated_trigrams / len(trigrams)) * 100.0

    return round(distinct_1, 4), round(distinct_2, 4), round(rep_rate, 2)


def run_sampling_experiment(
    models: Dict[str, nn.Module],
    vocab: Vocabulary,
    temperatures: List[float] = [0.2, 0.7, 1.2],
    seeds: List[str] = DEFAULT_SEEDS,
    max_new_tokens: int = 40,
    device: Optional[torch.device] = None,
    seq_len: int = 25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes comparative generation across models, temperatures, and seeds.
    Returns:
        samples_df: Full text generated for each combination
        diversity_df: Summary quantitative diversity metrics
    """
    sample_records = []
    diversity_records = []

    for model_name, model in models.items():
        for temp in temperatures:
            dist1_list, dist2_list, rep_list = [], [], []

            for s_idx, seed in enumerate(seeds, 1):
                gen_text = generate_text(
                    model=model,
                    vocab=vocab,
                    prompt=seed,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    device=device,
                    seq_len=seq_len,
                )
                d1, d2, rep = compute_diversity_metrics(gen_text)
                dist1_list.append(d1)
                dist2_list.append(d2)
                rep_list.append(rep)

                sample_records.append({
                    "Model": model_name,
                    "Temperature": temp,
                    "Seed ID": f"Seed {s_idx}",
                    "Seed Prompt": seed,
                    "Generated Text": gen_text,
                    "Distinct-1": d1,
                    "Distinct-2": d2,
                    "Repetition Rate (%)": rep,
                })

            diversity_records.append({
                "Model": model_name,
                "Temperature": temp,
                "Distinct-1": round(sum(dist1_list) / len(dist1_list), 4),
                "Distinct-2": round(sum(dist2_list) / len(dist2_list), 4),
                "Repetition Rate (%)": round(sum(rep_list) / len(rep_list), 2),
            })

    samples_df = pd.DataFrame(sample_records)
    diversity_df = pd.DataFrame(diversity_records)
    return samples_df, diversity_df
