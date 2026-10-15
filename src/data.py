"""
src/data.py
Module: Data Ingestion, Text Preprocessing, and Autoregressive Sequence Construction.

Context Acquisition:
- Responsibility: Ingestion, cleaning, tokenization, vocabulary mapping, and sliding-window dataset/DataLoader creation.
- Upstream: Project Gutenberg URL or local corpus file.
- Downstream: src/train.py, src/evaluate.py, src/generate.py, notebooks.
- Data Ownership: Raw text caching, vocabulary mappings (word2idx, idx2word), token tensors.
- Runtime Constraints: Execution time < 15s on CPU, RAM usage < 500 MB.
- Failure Expectations: Handle network failure via cache, reject empty/malformed text.
"""

from collections import Counter
import os
import re
from typing import Dict, List, Optional, Tuple
import requests
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATASET_URL = "https://www.gutenberg.org/files/1661/1661-0.txt"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LOCAL_CORPUS_PATH = os.path.join(DATA_DIR, "sherlock_holmes.txt")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def download_corpus(url: str = DEFAULT_DATASET_URL, target_path: str = LOCAL_CORPUS_PATH) -> str:
    """
    Downloads raw text corpus from Project Gutenberg if not already cached.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 10000:
        with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    print(f"Downloading corpus from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    text = response.text

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Corpus cached successfully to {target_path} ({len(text)} characters).")
    return text


def clean_gutenberg_text(text: str) -> str:
    """
    Strips Project Gutenberg license headers/footers and normalizes whitespace.
    """
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    ]

    start_idx = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            line_end = text.find("\n", pos)
            start_idx = line_end + 1 if line_end != -1 else pos + len(marker)
            break

    end_idx = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_idx = pos
            break

    cleaned = text[start_idx:end_idx].strip()
    return cleaned


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes text into words, contractions, and punctuation marks.
    Converts all tokens to lowercase for vocabulary consistency.
    """
    # Regex captures words with optional apostrophes (don't, holmes's) or individual punctuation
    token_pattern = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[.,!?;:\"()\-]")
    tokens = [tok.lower() for tok in token_pattern.findall(text)]
    return tokens


class Vocabulary:
    """
    Vocabulary class that manages bidirectional mapping between tokens and numerical indices.
    """
    def __init__(self, min_freq: int = 2, max_size: Optional[int] = 8000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.freqs: Counter = Counter()

        for idx, token in enumerate(SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def build_vocab(self, tokens: List[str]) -> "Vocabulary":
        self.freqs.update(tokens)
        candidate_words = [
            (word, count) for word, count in self.freqs.most_common()
            if count >= self.min_freq and word not in self.word2idx
        ]

        if self.max_size is not None:
            allowed_new_words = max(0, self.max_size - len(SPECIAL_TOKENS))
            candidate_words = candidate_words[:allowed_new_words]

        for word, _ in candidate_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        return self

    def __len__(self) -> int:
        return len(self.word2idx)

    def lookup_index(self, token: str) -> int:
        return self.word2idx.get(token, UNK_IDX)

    def lookup_token(self, idx: int) -> str:
        return self.idx2word.get(idx, UNK_TOKEN)

    def numericalize(self, tokens: List[str]) -> List[int]:
        return [self.lookup_index(t) for t in tokens]

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        words = []
        for idx in indices:
            tok = self.lookup_token(idx)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            words.append(tok)
        # Format tokens into readable text
        text = " ".join(words)
        # Attach punctuation to preceding word
        text = re.sub(r'\s+([.,!?;:\)])', r'\1', text)
        text = re.sub(r'(\()\s+', r'\1', text)
        return text


class AutoregressiveTextDataset(Dataset):
    """
    Constructs causal sliding-window input-target pairs for next-word prediction.
    Input: [w_t, w_{t+1}, ..., w_{t+seq_len-1}]
    Target: w_{t+seq_len}
    """
    def __init__(self, token_ids: List[int], seq_len: int = 25, step_size: int = 1):
        if len(token_ids) <= seq_len:
            raise ValueError(f"Token list length ({len(token_ids)}) must be > seq_len ({seq_len})")

        self.seq_len = seq_len
        self.inputs = []
        self.targets = []

        for i in range(0, len(token_ids) - seq_len, step_size):
            self.inputs.append(token_ids[i : i + seq_len])
            self.targets.append(token_ids[i + seq_len])

        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def prepare_dataset_and_dataloaders(
    seq_len: int = 25,
    step_size: int = 2,
    batch_size: int = 64,
    min_freq: int = 2,
    max_vocab_size: int = 6000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    url: str = DEFAULT_DATASET_URL,
    local_path: str = LOCAL_CORPUS_PATH,
) -> Tuple[Vocabulary, DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    End-to-end dataset preparation pipeline with temporal split.
    Returns:
        vocab, train_loader, val_loader, test_loader, metadata
    """
    raw_text = download_corpus(url=url, target_path=local_path)
    cleaned_text = clean_gutenberg_text(raw_text)
    tokens = tokenize_text(cleaned_text)

    if len(tokens) == 0:
        raise ValueError("Corpus tokenization yielded 0 tokens. Check cleaning logic.")

    vocab = Vocabulary(min_freq=min_freq, max_size=max_vocab_size)
    vocab.build_vocab(tokens)

    # Numericalize entire corpus
    token_ids = vocab.numericalize(tokens)
    total_tokens = len(token_ids)

    # Temporal / sequential split to prevent future data leakage
    train_end = int(total_tokens * train_ratio)
    val_end = int(total_tokens * (train_ratio + val_ratio))

    train_ids = token_ids[:train_end]
    val_ids = token_ids[train_end:val_end]
    test_ids = token_ids[val_end:]

    train_dataset = AutoregressiveTextDataset(train_ids, seq_len=seq_len, step_size=step_size)
    val_dataset = AutoregressiveTextDataset(val_ids, seq_len=seq_len, step_size=step_size)
    test_dataset = AutoregressiveTextDataset(test_ids, seq_len=seq_len, step_size=step_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "raw_characters": len(raw_text),
        "cleaned_characters": len(cleaned_text),
        "total_tokens": total_tokens,
        "vocab_size": len(vocab),
        "seq_len": seq_len,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }

    return vocab, train_loader, val_loader, test_loader, metadata
