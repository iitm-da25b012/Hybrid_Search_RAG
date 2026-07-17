from pathlib import Path

import fitz  # PyMuPDF
from bs4 import BeautifulSoup


class Parser:
    """
    Parses supported document types into plain text.

    Supported formats:
        - .txt
        - .md
        - .html
        - .htm
        - .pdf
    """

    @staticmethod
    def parse(file_path: str | Path):
        """
        Parse a file based on its extension.

        Returns:
            list[dict]

        Each dictionary contains:
        {
            "page": int,
            "text": str,
            "heading": str | None
        }
        """

        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == ".txt":
            return Parser._parse_txt(path)

        elif extension == ".md":
            return Parser._parse_markdown(path)

        elif extension in [".html", ".htm"]:
            return Parser._parse_html(path)

        elif extension == ".pdf":
            return Parser._parse_pdf(path)

        raise ValueError(f"Unsupported file type: {extension}")

    @staticmethod
    def _parse_txt(path: Path):
        text = path.read_text(encoding="utf-8")

        return [
            {
                "page": 1,
                "heading": None,
                "text": text,
            }
        ]

    @staticmethod
    def _parse_markdown(path: Path):
        text = path.read_text(encoding="utf-8")

        heading = None

        for line in text.splitlines():
            line = line.strip()

            if line.startswith("#"):
                heading = line.lstrip("#").strip()
                break

        return [
            {
                "page": 1,
                "heading": heading,
                "text": text,
            }
        ]

    @staticmethod
    def _parse_html(path: Path):
        html = path.read_text(encoding="utf-8")

        soup = BeautifulSoup(html, "html.parser")

        heading = None

        if soup.title:
            heading = soup.title.get_text(strip=True)

        text = soup.get_text(separator="\n")

        return [
            {
                "page": 1,
                "heading": heading,
                "text": text,
            }
        ]

    @staticmethod
    def _parse_pdf(path: Path):
        pdf = fitz.open(path)

        pages = []

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text()

            pages.append(
                {
                    "page": page_number,
                    "heading": None,
                    "text": text,
                }
            )

        pdf.close()

        return pages
