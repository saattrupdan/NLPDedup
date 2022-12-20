"""Datatypes used in the project."""

from typing import Any, Dict, Generator, Iterable, Union

CORPUS = Union[
    Iterable[str],
    Generator[str, None, None],
    Iterable[Dict[str, Any]],
    Generator[Dict[str, Any], None, None],
]
