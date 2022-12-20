"""Command line interface for deduplicating a text corpus."""

import click

from .deduper import Deduper


@click.command()
def main() -> None:
    """Deduplicate a text corpus."""
    pass


if __name__ == "__main__":
    main()
