from __future__ import annotations

from typing import List

import numpy as np


class Deduplicator:
    """
    Removes highly similar chunks using cosine similarity.

    If similarity >= threshold,
    only one chunk is kept.
    """

    def __init__(
        self,
        threshold: float = 0.95,
    ):

        self.threshold = threshold

    # -----------------------------------------------------

    def deduplicate(
        self,
        chunks,
        embeddings: List[List[float]],
    ):

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings must have equal length."
            )

        unique_chunks = []
        unique_embeddings = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):

            duplicate = False

            for existing in unique_embeddings:

                similarity = self.cosine_similarity(
                    embedding,
                    existing,
                )

                if similarity >= self.threshold:

                    duplicate = True

                    break

            if not duplicate:

                unique_chunks.append(chunk)

                unique_embeddings.append(embedding)

        return unique_chunks, unique_embeddings

    # -----------------------------------------------------

    @staticmethod
    def cosine_similarity(a, b):

        a = np.array(a)

        b = np.array(b)

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(a, b)
            / denominator
        )