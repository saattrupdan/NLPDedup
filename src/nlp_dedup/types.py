"""Datatypes used in the project."""

from typing import Any, Dict, Iterator, Protocol, Union


class Corpus(Protocol):
    """Protocol for a corpus."""

    def __iter__(self) -> Iterator[Union[str, Dict[str, Any]]]:
        """Iterate over the corpus."""
        ...

    def __next__(self) -> Union[str, Dict[str, Any]]:
        """Get the next item in the corpus."""
        ...
