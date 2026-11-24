"""Shared pieces of the semantic (headline) layer: the two embedding models,
keyword matching, and the per-topic classifier. Used by the calibration now
and by the daily pipeline later, so both treat headlines identically."""
from __future__ import annotations

import os
import re

import numpy as np


# ---------------------------------------------------------------- keywords
def compile_patterns(patterns: list[str]) -> re.Pattern | None:
    if not patterns:
        return None
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


class TopicKeywords:
    def __init__(self, spec: dict):
        self.include = compile_patterns(spec["keywords"])
        self.exclude = compile_patterns(spec.get("exclude") or [])

    def matches(self, titles: list[str]) -> np.ndarray:
        """True where a headline is a clear example of the topic."""
        # dtype=bool keeps this working on a day with no headlines (an empty list).
        hit = np.array([bool(self.include.search(t)) for t in titles], dtype=bool)
        if self.exclude is not None:
            hit &= ~np.array([bool(self.exclude.search(t)) for t in titles], dtype=bool)
        return hit

    def mask(self, title: str) -> str:
        """The headline with the topic's keywords removed, so the classifier
        has to learn from the surrounding words."""
        masked = self.include.sub(" ", title)
        return re.sub(r"\s+", " ", masked).strip() or "-"


# ---------------------------------------------------------------- models
def unit(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1)


class FakeEncoder:
    """Stand-in used only for testing offline (NEWSVOL_FAKE_MODELS=1): hashes
    words and character trigrams into a vector."""

    def __init__(self, dims: int):
        self.dims = dims

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            t = f"  {text.lower()}  "
            for i in range(len(t) - 2):
                out[row, hash(t[i:i + 3]) % self.dims] += 0.3
            for word in text.lower().split():
                out[row, hash(word) % self.dims] += 1
        return out


class ScreenModel:
    """Fast static model: used to assign topics."""

    def __init__(self, name: str):
        if os.environ.get("NEWSVOL_FAKE_MODELS"):
            self.model = FakeEncoder(256)
        else:
            from model2vec import StaticModel
            self.model = StaticModel.from_pretrained(name, force_download=False)

    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(self.model, FakeEncoder):
            return unit(self.model.encode(texts))
        return unit(self.model.encode(list(texts), batch_size=2048))


class ScoreModel:
    """Slower transformer: used to measure tone."""

    def __init__(self, name: str, prefix: str):
        self.prefix = prefix
        if os.environ.get("NEWSVOL_FAKE_MODELS"):
            self.model = FakeEncoder(384)
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = [self.prefix + t for t in texts]
        if isinstance(self.model, FakeEncoder):
            return unit(self.model.encode(texts))
        return unit(self.model.encode(texts, batch_size=128))


def centered(vectors: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Remove the direction every headline shares ("this is a news headline"),
    so that similarities reflect what distinguishes headlines from each other."""
    return unit(vectors - mean)


def tone_scores(vectors: np.ndarray, calm: np.ndarray, stress: np.ndarray) -> np.ndarray:
    """Positive: closer to the stress sentences; negative: closer to the calm ones.
    All inputs must be centered and unit length."""
    return (vectors @ stress.T).mean(axis=1) - (vectors @ calm.T).mean(axis=1)
