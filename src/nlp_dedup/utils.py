"""Utility functions used in other modules."""

import re
from unicodedata import normalize


def default_normalization(doc: str) -> str:
    """NFKC normalise document and remove punctuation

    Args:
        doc (str):
            The document to normalize.

    Returns:
        str:
            The normalized document.
    """
    doc = normalize("NFKC", doc)
    doc = re.sub(r"[\.\,\:\;\!\?\(\)\[\]\{\}]", " ", doc)
    return re.sub(" +", " ", doc)
