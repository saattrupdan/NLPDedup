# NLPDedup

Remove duplicates and near-duplicates from text corpora, no matter the scale.

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://saattrupdan.github.io/NLPDedup/nlp_dedup.html)
[![License](https://img.shields.io/github/license/saattrupdan/NLPDedup)](https://github.com/saattrupdan/NLPDedup/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/saattrupdan/NLPDedup)](https://github.com/saattrupdan/NLPDedup/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/saattrupdan/NLPDedup/tree/main/tests)


Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
- Kenneth Enevoldsen (kennethcenevoldsen@gmail.com)


# Installation

The package is available on PyPI, so you can install the package using your favourite
package manager. For instance, `pip install nlp_dedup` or `poetry add nlp_dedup`.


# Quick start

If the corpus is stored as `corpus.txt` (both `txt` and `jsonl` files are supported),
the following deduplicates the corpus and stores the deduplicates corpus into the
folder `deduplicated`:

```
$ dedup corpus.txt deduplicated
```

This defaults to deduplicating based on blocks of 13 consecutive words, where two
documents are considered near-duplicate if they have more than 80% of these blocks in
common. This can all be changed to your specific needs, however.

See `$ dedup --help` for more information on all the settings.
