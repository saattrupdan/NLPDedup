"""Utility functions used in other modules."""

import json
import re
from pathlib import Path
from typing import List, Union
from unicodedata import normalize

from datasketch import LeanMinHash, MinHash


def append_to_jsonl(output_path: Union[str, Path], **data_kwargs):
    """Appends the document to a JSONL file.

    Args:
        output_path (str or Path):
            The name of the JSONL file to append to.
        **data_kwargs:
            The keyword arguments to append to the JSONL file.
    """
    with Path(output_path).open("a") as f:
        jsonned = json.dumps(data_kwargs)
        f.write(jsonned + "\n")


def get_minhash(
    doc: str,
    split_method: str,
    ngram_size: int,
    ngram_stride: int,
    num_minhashes: int,
    random_seed: int,
) -> LeanMinHash:
    """Returns a minhash fingerprint for the given document.

    Args:
        doc (str):
            The document to create the MinHash object for.
        split_method (str):
            The method to split the document into shingles.
            Can be 'word_ngram', 'paragraph', 'none' or None.
        ngram_size (int):
            The size of the ngrams to use.
        ngram_stride (int):
            The stride of the ngrams to use.
        num_minhashes (int):
            The number of minhashes to use.
        random_seed (int):
            The random seed to use.

    Returns:
        LeanMinHash:
            The minhash fingerprint for the given document.

    Raises:
        ValueError:
            If `split_method` is not 'word_ngram', 'paragraph', 'none' or None.
    """
    # Extract shingles from the document, depending on the `split_method`
    shingles = get_shingles(
        doc,
        split_method=split_method,
        ngram_size=ngram_size,
        ngram_stride=ngram_stride,
    )

    # Initialise the fingerprint
    minhash = MinHash(num_perm=num_minhashes, seed=random_seed)

    # Add all the shingles to the fingerprint
    minhash.update_batch([shingle.encode("utf-8") for shingle in shingles])

    # Convert the fingerprint to a LeanMinHash fingerprint, to save memory and increase
    # performance
    minhash = LeanMinHash(minhash, seed=random_seed)

    # Return the fingerprint
    return minhash


def get_shingles(
    doc: str,
    split_method: str,
    ngram_size: int,
    ngram_stride: int,
) -> List[str]:
    """Extracts the shingles from a document.

    Args:
        doc (str):
            The document to extract the shingles from.
        split_method (str):
            The method to split the document into shingles. Can be 'word_ngram',
            'paragraph', 'none' or None.
        ngram_size (int):
            The size of the ngrams to use.
        ngram_stride (int):
            The stride of the ngrams to use.

    Returns:
        list of str:
            The shingles extracted from the document.

    Raises:
        ValueError:
            If `split_method` is not 'word_ngram', 'paragraph', 'none' or None.
    """
    # Normalise document
    doc = normalize("NFKC", doc)
    doc = re.sub(r"[\.\,\:\;\!\?\(\)\[\]\{\}]", " ", doc)
    doc = re.sub(" +", " ", doc)

    # Extract shingles from the document, depending on the `split_method`
    if split_method == "word_ngram":
        words = [word for word in doc.split(" ") if len(word) > 0]
        max_word_idx = 1 + len(words) - ngram_size
        shingles = [
            " ".join(words[i : i + ngram_size]).strip()
            for i in range(0, max_word_idx, ngram_stride)
        ] or [doc]
    elif split_method == "paragraph":
        shingles = [p for p in doc.split("\n") if len(p) > 0]
    elif split_method == "none" or split_method is None:
        shingles = [doc]
    else:
        raise ValueError(f"Invalid split method: {split_method}")

    return shingles
