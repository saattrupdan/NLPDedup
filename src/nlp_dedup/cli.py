"""Command line interface for deduplicating a text corpus."""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Generator, Union

import click

from .deduper import Deduper


@click.command()
@click.argument("corpus", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path())
@click.option(
    "--split-method",
    type=click.Choice(["word_ngram", "paragraph", "none"]),
    default="word_ngram",
    show_default=True,
    help="The method to split the documents into shingles.",
)
@click.option(
    "--ngram-size",
    type=int,
    default=13,
    show_default=True,
    help="The size of the ngrams to use for the word_ngram split method.",
)
@click.option(
    "--ngram-stride",
    type=int,
    default=1,
    show_default=True,
    help="The stride of the ngrams to use for the word_ngram split method.",
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="The similarity threshold to use for the deduplication.",
)
@click.option(
    "--num-minhashes",
    type=int,
    default=128,
    show_default=True,
    help="The number of minhashes to use for the deduplication.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1_000_000,
    show_default=True,
    help="The number of documents to process at once.",
)
@click.option(
    "--n-jobs",
    type=int,
    default=-1,
    show_default=True,
    help="The number of jobs to use for the deduplication.",
)
@click.option(
    "--random-seed",
    type=int,
    default=4242,
    show_default=True,
    help="The random seed to use for the deduplication.",
)
@click.option(
    "--store-corpus-to-disk/--no-store-corpus-to-disk",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to store the corpus to disk.",
)
@click.option(
    "--store-mask-to-disk/--no-store-mask-to-disk",
    type=bool,
    default=False,
    show_default=True,
    help="Whether to store the mask to disk.",
)
@click.option(
    "--store-lsh-cache-to-disk/--no-store-lsh-cache-to-disk",
    type=bool,
    default=False,
    show_default=True,
    help="Whether to store the LSH cache to disk.",
)
@click.option(
    "--store-config-to-disk/--no-store-config-to-disk",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to store the config to disk.",
)
@click.option(
    "--verbose/--no-verbose",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to print output.",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    show_default=True,
    help="""The name of the column containing the text, if the entries in the corpus
    are dictionaries.""",
)
@click.option(
    "--overwrite/--no-overwrite",
    type=bool,
    default=False,
    show_default=True,
    help="Whether to overwrite the output directory if it already exists.",
)
def main(
    corpus: str,
    split_method: str,
    ngram_size: int,
    ngram_stride: int,
    similarity_threshold: float,
    num_minhashes: int,
    batch_size: int,
    n_jobs: int,
    random_seed: int,
    store_corpus_to_disk: bool,
    store_mask_to_disk: bool,
    store_lsh_cache_to_disk: bool,
    store_config_to_disk: bool,
    verbose: bool,
    text_column: str,
    output_dir: str,
    overwrite: bool,
) -> None:
    """Deduplicate a text corpus.

    Args:
        corpus (str):
            The path to the file containing the text corpus to deduplicate.
        split_method (str):
            The method to split the documents into shingles.
        ngram_size (int):
            The size of the ngrams to use for the word_ngram split method.
        ngram_stride (int):
            The stride of the ngrams to use for the word_ngram split method.
        similarity_threshold (float):
            The similarity threshold to use for the deduplication.
        num_minhashes (int):
            The number of minhashes to use for the deduplication.
        batch_size (int):
            The number of documents to process at once.
        n_jobs (int):
            The number of jobs to use for the deduplication.
        random_seed (int):
            The random seed to use for the deduplication.
        store_corpus_to_disk (bool):
            Whether to store the corpus to disk.
        store_mask_to_disk (bool):
            Whether to store the mask to disk.
        store_lsh_cache_to_disk (bool):
            Whether to store the LSH cache to disk.
        store_config_to_disk (bool):
            Whether to store the config to disk.
        verbose (bool):
            Whether to print output.
        text_column (str):
            The name of the column containing the text, if the entries in the corpus
            are dictionaries.
        output_dir (str):
            The directory to store the deduplicated corpus in.
        overwrite (bool):
            Whether to overwrite the output directory if it already exists.
    """
    # Initialise the Deduper
    deduper = Deduper(
        split_method=split_method,
        ngram_size=ngram_size,
        ngram_stride=ngram_stride,
        similarity_threshold=similarity_threshold,
        num_minhashes=num_minhashes,
        batch_size=batch_size,
        n_jobs=n_jobs,
        random_seed=random_seed,
        store_corpus_to_disk=store_corpus_to_disk,
        store_mask_to_disk=store_mask_to_disk,
        store_lsh_cache_to_disk=store_lsh_cache_to_disk,
        store_config_to_disk=store_config_to_disk,
        verbose=verbose,
    )

    # Count the number of lines in the corpus
    proc = subprocess.Popen(
        ["wc", "-l", corpus], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    num_docs = int(proc.communicate()[0].split()[0])

    # Create generator for the corpus
    def corpus_generator() -> Generator[Union[str, Dict[str, Any]], None, None]:
        with Path(corpus).open() as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield line.strip("\n")

    # Deduplicate the corpus
    deduper.deduplicate(
        corpus=corpus_generator(),
        text_column=text_column,
        output_dir=output_dir,
        overwrite=overwrite,
        num_docs=num_docs,
    )
