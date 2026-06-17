from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

from .settings import FASTEMBED_MODEL


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class DeterministicEmbedder:
    """Small hashing embedder for tests and offline smoke runs."""

    dimension: int = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in text.lower().replace(".", " ").replace(",", " ").split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:2], "big") % self.dimension
                sign = 1.0 if digest[2] % 2 == 0 else -1.0
                vector[index] += sign
            vectors.append(_normalize(vector))
        return vectors


class FastEmbedder:
    def __init__(self, model_name: str = FASTEMBED_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        sample = list(self._model.embed(["dimension probe"]))[0]
        self.dimension = len(sample)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(value) for value in vector] for vector in self._model.embed(texts)]


def get_embedder(backend: str = "fastembed") -> Embedder:
    if backend == "deterministic":
        return DeterministicEmbedder()
    if backend != "fastembed":
        raise ValueError(f"unknown embedding backend: {backend}")
    return FastEmbedder()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
