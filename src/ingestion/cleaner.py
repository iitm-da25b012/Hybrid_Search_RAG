import re
import unicodedata


class Cleaner:
    """
    Cleans and normalizes extracted document text.
    """

    @staticmethod
    def clean(text: str) -> str:
        """
        Apply all cleaning steps.

        Parameters
        ----------
        text : str
            Raw extracted text.

        Returns
        -------
        str
            Cleaned text.
        """

        if not text:
            return ""

        text = Cleaner.normalize_unicode(text)
        text = Cleaner.normalize_newlines(text)
        text = Cleaner.remove_extra_spaces(text)
        text = Cleaner.remove_extra_blank_lines(text)
        text = Cleaner.remove_control_characters(text)

        return text.strip()

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize unicode characters.

        Example:
            ﬁ -> fi
            “ ” -> "
        """
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def normalize_newlines(text: str) -> str:
        """
        Convert all newline styles to Unix format.
        """

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        return text

    @staticmethod
    def remove_extra_spaces(text: str) -> str:
        """
        Remove trailing spaces and repeated spaces.
        """

        cleaned_lines = []

        for line in text.split("\n"):
            line = re.sub(r"[ \t]+", " ", line)
            cleaned_lines.append(line.strip())

        return "\n".join(cleaned_lines)

    @staticmethod
    def remove_extra_blank_lines(text: str) -> str:
        """
        Replace multiple blank lines with a single blank line.
        """

        return re.sub(r"\n{3,}", "\n\n", text)

    @staticmethod
    def remove_control_characters(text: str) -> str:
        """
        Remove non-printable control characters while preserving
        newlines and tabs.
        """

        return "".join(
            char
            for char in text
            if char == "\n"
            or char == "\t"
            or char.isprintable()
        )

    @staticmethod
    def collapse_whitespace(text: str) -> str:
        """
        Optional utility.

        Converts all whitespace into single spaces.
        Useful before embedding if needed.
        """

        return re.sub(r"\s+", " ", text).strip()