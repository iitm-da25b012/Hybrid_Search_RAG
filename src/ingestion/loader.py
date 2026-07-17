from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .cleaner import Cleaner
from .metadata import MetadataBuilder
from .parser import Parser


@dataclass
class Document:
    """
    Standard document object used throughout the pipeline.
    """

    text: str
    metadata: dict


class DocumentLoader:
    """
    Loads a document, processes it, and returns Document objects.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load(self, file_path: str | Path) -> list[Document]:
        """
        Complete ingestion pipeline.

        Parameters
        ----------
        file_path : str | Path

        Returns
        -------
        list[Document]
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

        # -------------------------------------------------
        # Store original document
        # -------------------------------------------------

        destination = self.raw_dir / path.name

        if not destination.exists():
            shutil.copy2(path, destination)

        # -------------------------------------------------
        # Parse document
        # -------------------------------------------------

        parsed_pages = Parser.parse(path)

        documents = []

        for page in parsed_pages:

            clean_text = Cleaner.clean(page["text"])

            metadata = MetadataBuilder.build(
                source=path.name,
                extension=path.suffix.lower(),
                page=page["page"],
                heading=page["heading"],
            )

            documents.append(
                Document(
                    text=clean_text,
                    metadata=metadata,
                )
            )

        # -------------------------------------------------
        # Save processed JSON
        # -------------------------------------------------

        self._save_processed(path.stem, documents)

        return documents

    def _save_processed(
        self,
        filename: str,
        documents: list[Document],
    ) -> None:
        """
        Save processed documents to JSON.
        """

        output_file = self.processed_dir / f"{filename}.json"

        data = []

        for document in documents:
            data.append(asdict(document))

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False,
            )