"""Class that deduplicates a corpus.

The deduplication is based on the MinHash algorithm [1], which is optimised using
parallelism and vectorisation.

Authors:
    - Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
    - Kenneth Enevoldsen (kennethcenevoldsen@gmail.com)

References:
    [1] Broder, Andrei Z. "On the resemblance and containment of documents."
        Proceedings. Compression and Complexity of SEQUENCES 1997
        (Cat. No. 97TB100171). IEEE, 1997.
"""

import itertools as it
import json
import logging
import multiprocessing as mp
import pickle
import shutil
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import more_itertools as mit
from datasketch import MinHashLSH
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .types import Corpus
from .utils import append_to_jsonl, get_minhash

# Set up logging
logger = logging.getLogger(__name__)


class Deduper:
    """Class that deduplicates an iterable corpus.

    The deduplication is based on the MinHash algorithm [1].

    Args:
        split_method (str or None, optional):
            The method to split the documents into shingles. Can be either
            'word_ngram', 'paragraph', 'none' or None. Here 'none' or None
            means that a document is not split up at all. Defaults to
            'word_ngram'.
        ngram_size (int, optional):
            The size of the ngram shingles. Only relevant if `split_method` is
            'word_ngram'. Defaults to 13.
        ngram_stride (int, optional):
            The stride of the ngram shingles. Only relevant if `split_method`
            is 'word_ngram'. Defaults to 1, corresponding to no stride.
        similarity_threshold (float, optional):
            The similarity threshold to use for the MinHash functions.
            Defaults to 0.8.
        num_minhashes (int, optional):
            The number of MinHash functions to use. Defaults to 128.
        batch_size (int, optional):
            The number of documents to process at a time. Defaults to 100,000,000.
        n_jobs (int, optional):
            The number of parallel jobs to use. If set to -1 then all available
            cores are used. Defaults to -1.
        random_seed (int, optional):
            The random seed to use for the MinHash functions. Defaults to 4242.
        normalization_func: (Callable[[str], str], optional):
            The function used to normalize documents before they are compared to
            ignore insignificant differences. Needs to be pickleable.
        store_corpus_to_disk (bool, optional):
            Whether to store the corpus to disk. Defaults to True.
        store_mask_to_disk (bool, optional):
            Whether to store the mask to disk. Defaults to True.
        store_lsh_cache_to_disk (bool, optional):
            Whether to store the LSH cache to disk. Defaults to True.
        store_config_to_disk (bool, optional):
            Whether to store the configuration to disk. Defaults to True.
        return_generator (bool, optional):
            Whether to return a generator which yields the mask. Defaults to False.
        verbose (bool, optional):
            Print progress to stdout. Defaults to True.

    Attributes:
        split_method (str): The splitting method for extracting shingles.
        ngram_size (str): The size of the ngram shingles.
        ngram_stride (str): The stride used for the ngram shingles.
        similarity_threshold (float): The Jaccard similarity threshold.
        num_minhashes (int): The number of MinHash functions to use.
        batch_size (int): The number of documents to process at a time.
        n_jobs (int): The number of parallel jobs to use.
        random_seed (int): The random seed to use for the MinHash functions.
        normalization_func (Callable): The function used for normalization.
        store_corpus_to_disk (bool): Whether to store the corpus to disk.
        store_mask_to_disk (bool): Whether to store the mask to disk.
        store_lsh_cache_to_disk (bool): Whether to store the LSH cache to disk.
        store_config_to_disk (bool): Whether to store the configuration to disk.
        return_generator (bool): Whether to return a generator which yields the mask.
        verbose (bool): Print progress to stdout.

    References:
        [1] Broder, Andrei Z. "On the resemblance and containment of documents."
            Proceedings. Compression and Complexity of SEQUENCES 1997
            (Cat. No. 97TB100171). IEEE, 1997.
    """

    def __init__(
        self,
        split_method: Optional[str] = "word_ngram",
        ngram_size: int = 13,
        ngram_stride: int = 1,
        similarity_threshold: float = 0.8,
        num_minhashes: int = 128,
        batch_size: int = 100_000_000,
        n_jobs: int = -1,
        random_seed: int = 4242,
        save_mask: bool = True,
        store_corpus_to_disk: bool = True,
        store_mask_to_disk: bool = True,
        store_lsh_cache_to_disk: bool = True,
        store_config_to_disk: bool = True,
        return_generator: bool = False,
        verbose: bool = True,
    ) -> None:

        # Initialise attributes
        self.split_method = "none" if split_method is None else split_method
        self.ngram_size = ngram_size
        self.ngram_stride = ngram_stride
        self.similarity_threshold = similarity_threshold
        self.num_minhashes = num_minhashes
        self.batch_size = batch_size
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.random_seed = random_seed
        self.save_mask = save_mask
        self.store_corpus_to_disk = store_corpus_to_disk
        self.store_mask_to_disk = store_mask_to_disk
        self.store_lsh_cache_to_disk = store_lsh_cache_to_disk
        self.store_config_to_disk = store_config_to_disk
        self.return_generator = return_generator
        self.verbose = verbose

        # Initialise mask if we are saving it
        if save_mask:
            self.mask: List[Dict[str, Union[int, bool]]] = list()

        # Initialise the LSH cache
        self.lsh_cache = MinHashLSH(
            threshold=self.similarity_threshold, num_perm=self.num_minhashes
        )

    def deduplicate(
        self,
        corpus: Corpus,
        text_column: str = "text",
        output_dir: Union[str, Path] = "deduplicated",
        overwrite: bool = False,
        num_docs: Optional[int] = None,
    ) -> Union[Iterable, None]:
        """Removes duplicate documents from the corpus.

        Args:
            corpus (iterable or generator of strings or dictionaries):
                The corpus to deduplicate.
            text_column (str, optional):
                The name of the column in the corpus that contains the document
                texts. Defaults to 'text'.
            output_dir (str or Path, optional):
                The name of the output directory. Defaults to 'deduplicated'.
            overwrite (bool, optional):
                Whether to overwrite the output file if it already exists. Defaults to
                False.
            num_docs (int, optional):
                The number of documents in the corpus. Defaults to None.

        Returns:
            Iterable or None:
                If `return_generator` is True, then a generator which yields a
                dictionary with keys `id` and `duplicate`. Otherwise, None.

        Raises:
            FileExistsError:
                If the output file already exists and `overwrite` is False.
        """
        iterable = self._deduplicate(
            corpus=corpus,
            text_column=text_column,
            output_dir=output_dir,
            overwrite=overwrite,
            num_docs=num_docs,
        )
        if self.return_generator:
            return iterable
        else:
            for _ in iterable:
                pass
            return None

    def _deduplicate(  # noqa: C901
        self,
        corpus: Corpus,
        text_column: str = "text",
        output_dir: Union[str, Path] = "deduplicated",
        overwrite: bool = False,
        num_docs: Optional[int] = None,
    ) -> Iterable:
        """Helper function for the `deduplicate` method.

        Args:
            corpus (iterable or generator of strings or dictionaries):
                The corpus to deduplicate.
            text_column (str, optional):
                The name of the column in the corpus that contains the document
                texts. Defaults to 'text'.
            output_dir (str or Path, optional):
                The name of the output directory. Defaults to 'deduplicated'.
            overwrite (bool, optional):
                Whether to overwrite the output file if it already exists. Defaults to
                False.
            num_docs (int, optional):
                The number of documents in the corpus. Defaults to None.

        Yields:
            dict or None:
                A dictionary with keys `id` and `duplicate` if `return_generator` is
                True, and otherwise None.

        Raises:
            FileExistsError:
                If the output file already exists and `overwrite` is False.
        """
        # Register number of documents in the corpus, if possible
        try:
            num_docs = len(corpus)  # type: ignore[arg-type]
        except TypeError:
            pass

        # Get a sample of the corpus
        corpus = iter(corpus)
        sample = next(corpus)

        # Add the first element back in
        corpus = it.chain([sample], corpus)

        # If the corpus contains dictionaries then convert it to an iterable of strings
        if isinstance(sample, dict):
            corpus = (sample[text_column] for sample in corpus)  # type: ignore[index]

        # Ensure that `output_dir` is a Path object
        output_dir = Path(output_dir)

        # If the output directory already exists then save to disk, which either
        # overwrites the existing files or raises an error, depending on whether we are
        # overwriting or not
        if output_dir.exists():
            self.save_to_disk(directory=output_dir, overwrite=overwrite)

        # Else, if the output directory doesn't exist and we are storing anything to
        # disk, then create the directory
        elif (
            self.store_corpus_to_disk
            or self.store_lsh_cache_to_disk
            or self.store_mask_to_disk
            or self.store_config_to_disk
        ):
            output_dir.mkdir(parents=True)

        # Set up paths
        output_path = output_dir / "deduplicated_corpus.jsonl"
        mask_path = output_dir / "mask.jsonl"
        lsh_cache_path = output_dir / "lsh_cache.pkl"
        config_path = output_dir / "config.json"

        # Store the deduper config to disk
        if self.store_config_to_disk:
            with config_path.open("w") as f:
                json.dump(self.config, f, indent=4)

        #  Split the corpus into batches of `self.batch_size` documents
        batches = mit.ichunked(corpus, self.batch_size)

        # Initialise the counting variables
        duplicates = 0
        num_processed = 0
        idx = 0

        # Initialise the main progress bar
        with tqdm(
            iterable=batches,
            desc="Deduplicating",
            total=num_docs,
            disable=(not self.verbose),
            leave=False,
        ) as pbar:

            # Define the function that will be called in parallel
            fn = delayed(
                partial(
                    get_minhash,
                    split_method=self.split_method,
                    ngram_size=self.ngram_size,
                    ngram_stride=self.ngram_stride,
                    num_minhashes=self.num_minhashes,
                    random_seed=self.random_seed,
                )
            )

            # Iterate over the batches
            for batch in pbar:

                # Create a copy of the batch to ensure that we're not modifying the
                # original
                batch, batch_copy = it.tee(batch)

                # Compute size of the batch
                new_num_processed = num_processed + self.batch_size
                if num_docs is not None:
                    new_num_processed = min(new_num_processed, num_docs)
                batch_size = new_num_processed - num_processed

                # Compute the fingerprint for the document
                with tqdm(
                    iterable=batch,
                    disable=(not self.verbose),
                    leave=False,
                    desc="Computing minhashes",
                    total=batch_size,
                ) as batch_pbar:
                    with Parallel(n_jobs=self.n_jobs, timeout=1000) as parallel:
                        minhashes = parallel(fn(doc) for doc in batch_pbar)

                # If there were no elements in the batch then we skip it
                if minhashes is None:
                    continue

                # Iterate over the minhashes, deduplicating the documents
                with tqdm(
                    iterable=batch_copy,
                    disable=(not self.verbose),
                    leave=False,
                    desc="Deduplicating batch",
                    total=batch_size,
                ) as batch_pbar:
                    for doc, minhash in zip(batch_pbar, minhashes):

                        # Compute list of documents that are near-duplicates of the
                        # current document
                        candidates = self.lsh_cache.query(minhash)

                        # If there are not any near-duplicate candidates then store the
                        # document in the LSH cache and append it to the JSONL output
                        if len(candidates) == 0:

                            # Insert the document into the LSH cache
                            self.lsh_cache.insert(idx, minhash)

                            # Store the non-duplicate document in the JSONL output
                            if self.store_corpus_to_disk:
                                append_to_jsonl(
                                    id=idx, text=doc, output_path=output_path
                                )

                            # Compute the mask for the document
                            mask_entry = dict(id=idx, duplicate=False)

                        # Otherwise, increment the number of duplicate documents
                        else:
                            duplicates += 1

                            # Compute the mask for the document
                            mask_entry = dict(id=idx, duplicate=True)

                        # Add the mask to the mask attribute
                        if self.save_mask:
                            self.mask.append(mask_entry)

                        # Yield the mask
                        if self.return_generator:
                            yield mask_entry

                        # Store the mask to disk
                        if self.store_mask_to_disk:
                            append_to_jsonl(output_path=mask_path, **mask_entry)

                        # Update the document index
                        idx += 1

                # Store the LSH cache to disk
                if self.store_lsh_cache_to_disk:
                    with lsh_cache_path.open("wb") as f:
                        pickle.dump(self.lsh_cache, f)

                # Update the number of documents processed, and compute the number of
                # documents in the batch
                num_processed = new_num_processed

                # Update the progress bar
                pbar.update(batch_size)
                pbar.set_description(
                    f"Deduplicating - {duplicates / num_processed:.2%} "
                    "near-duplicates found"
                )

        # Log final update
        if self.verbose:
            logger.info("Finished deduplicating corpus.")
            logger.info(f"- {num_processed:,} documents processed.")
            logger.info(
                f"- {duplicates / num_processed:.2%} documents marked as duplicates."
            )

    def save_to_disk(
        self,
        directory: Union[str, Path],
        overwrite: bool = False,
    ) -> None:
        """Save the Deduper to disk.

        Args:
            directory (str or Path, optional):
                The name of the output directory.
            overwrite (bool, optional):
                Whether to overwrite the output file if it already exists. Defaults to
                False.
        """
        # Ensure that `directory` is a Path object
        directory = Path(directory)

        # If the output file already exists then raise an error if `overwrite`
        # is False and otherwise delete the file
        if directory.exists() and not overwrite:
            raise FileExistsError(
                f"Output directory {directory!r} already exists. Please set "
                "`overwrite` to True to overwrite the files. If you are loading an "
                "existing Deduper from the directory then the previous config, mask "
                "and LSH cache will still will not be lost and will be stored in the "
                "directory."
            )

        # Delete the output directory if `overwrite` is set
        elif directory.exists() and overwrite:
            shutil.rmtree(directory)

        # Create the output directory
        directory.mkdir(parents=True, exist_ok=True)

        # Store existing mask
        if self.save_mask and self.store_mask_to_disk:
            mask_path = directory / "mask.jsonl"
            mask_str = "\n".join(json.dumps(sample) for sample in self.mask)
            with mask_path.open("w") as f:
                f.write(mask_str)

        # Store existing LSH cache
        if self.store_lsh_cache_to_disk:
            lsh_cache_path = directory / "lsh_cache.pkl"
            with lsh_cache_path.open("wb") as f:
                pickle.dump(self.lsh_cache, f)

        # Store existing configuration
        if self.store_config_to_disk:
            config_path = directory / "config.json"
            with config_path.open("w") as f:
                json.dump(self.config, f, indent=4)

    @classmethod
    def load_from_disk(cls, directory: Union[str, Path]) -> "Deduper":
        """Load a Deduper from disk.

        Args:
            directory (str or Path):
                The directory to load the Deduper from.

        Returns:
            Deduper:
                The Deduper loaded from disk.

        Raises:
            FileNotFoundError:
                If the directory does not exist.
        """
        # Ensure that `directory` is a Path
        directory = Path(directory)

        # Check if the directory exists, and raise an error if it doesn't
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")

        # Load the config file
        config_path = directory / "config.json"
        with config_path.open() as f:
            config = json.load(f)

        # Create the Deduper
        deduper = cls(**config)

        # Load the mask if it exists
        mask_path = directory / "mask.jsonl"
        if mask_path.exists():
            with mask_path.open() as f:
                mask = [json.loads(line) for line in f]
            deduper.mask = mask

        # Load the LSH cache
        lsh_cache_path = directory / "lsh_cache.pkl"
        with lsh_cache_path.open("rb") as f:
            deduper.lsh_cache = pickle.load(f)

        return deduper

    def reset(self):
        """Reset the deduplicator, removing the mask and the LSH cache"""
        if self.save_mask:
            self.mask = list()
        self.lsh_cache = MinHashLSH(
            threshold=self.similarity_threshold, num_perm=self.num_minhashes
        )
        return self

    @property
    def config(self) -> dict:
        """Get the configuration of the deduplicator.

        Returns:
            dict:
                The configuration of the deduplicator.
        """
        return dict(
            split_method=self.split_method,
            ngram_size=self.ngram_size,
            ngram_stride=self.ngram_stride,
            similarity_threshold=self.similarity_threshold,
            num_minhashes=self.num_minhashes,
            batch_size=self.batch_size,
            n_jobs=self.n_jobs,
            random_seed=self.random_seed,
            verbose=self.verbose,
        )
