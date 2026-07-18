from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import yaml

from .fixed import FixedChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker


class ChunkManager:
    """
    Controls which chunking strategy is used.

    Reads configuration from configs/chunking.yaml.
    """

    def __init__(
        self,
        config_path: str = "configs/chunking.yaml",
        output_dir: str = "data/chunks",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.strategy = self.config["strategy"]

        self.chunker = self._build_chunker()

    # -----------------------------------------------------

    def _build_chunker(self):

        if self.strategy == "fixed":

            cfg = self.config["fixed"]

            return FixedChunker(
                chunk_size=cfg["chunk_size"],
                overlap=cfg["overlap"],
            )

        elif self.strategy == "recursive":

            cfg = self.config["recursive"]

            return RecursiveChunker(
                chunk_size=cfg["chunk_size"],
                overlap=cfg["overlap"],
            )

        elif self.strategy == "semantic":

            cfg = self.config["semantic"]

            return SemanticChunker(
                similarity_threshold=cfg["similarity_threshold"],
                min_chunk_size=cfg["min_chunk_size"],
                max_chunk_size=cfg["max_chunk_size"],
            )

        raise ValueError(
            f"Unknown chunking strategy: {self.strategy}"
        )

    # -----------------------------------------------------

    def chunk_documents(
        self,
        documents: List,
        output_name: str,
    ):
        """
        Chunk documents using the configured strategy.
        """

        chunks = self.chunker.chunk_documents(documents)

        self.save_chunks(
            chunks,
            output_name,
        )

        return chunks

    # -----------------------------------------------------

    def save_chunks(
        self,
        chunks,
        output_name: str,
    ):

        output_file = (
            self.output_dir
            / f"{output_name}_{self.strategy}.json"
        )

        data = [
            asdict(chunk)
            for chunk in chunks
        ]

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False,
            )