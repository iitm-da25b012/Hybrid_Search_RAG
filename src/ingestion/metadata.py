from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path


class MetadataBuilder:
    """
    Builds standardized metadata for every document.

    Required fields:
        - source
        - page
        - heading
        - extension

    Additional fields are included to make the pipeline
    more production-ready.
    """

    @staticmethod
    def build(
        source: str,
        extension: str,
        page: int = 1,
        heading: str | None = None,
    ) -> dict:
        """
        Build metadata dictionary.

        Parameters
        ----------
        source : str
            Original filename.

        extension : str
            File extension.

        page : int
            Page number.

        heading : str | None
            Section heading if available.

        Returns
        -------
        dict
        """

        return {
            "source": source,
            "page": page,
            "heading": heading,
            "extension": extension,
            "document_id": MetadataBuilder.generate_document_id(source),
            "created_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def generate_document_id(source: str) -> str:
        """
        Generate a stable document ID from the filename.

        Example:
            guide.pdf
            ↓
            8d72c4f3e7...
        """

        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    @staticmethod
    def extract_file_info(file_path: str | Path) -> dict:
        """
        Extract useful file information.

        This method is optional and can be used later
        in the ingestion pipeline.
        """

        path = Path(file_path)

        stats = path.stat()

        return {
            "filename": path.name,
            "stem": path.stem,
            "extension": path.suffix.lower(),
            "size_bytes": stats.st_size,
            "last_modified": datetime.fromtimestamp(
                stats.st_mtime
            ).isoformat(),
        }