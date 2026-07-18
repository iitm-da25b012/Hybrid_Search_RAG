from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Generates embeddings using a local SentenceTransformer model.

    Default model:
        BAAI/bge-small-en-v1.5

    Embedding Dimension:
        384
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):

        self.model = SentenceTransformer(model_name)

    # -----------------------------------------------------

    def embed(
        self,
        text: str,
    ) -> List[float]:

        if not text.strip():
            return []

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    # -----------------------------------------------------

    def embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        if len(texts) == 0:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()