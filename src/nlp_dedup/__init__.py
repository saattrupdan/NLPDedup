"""
.. include:: ../../README.md
"""

import pkg_resources

from .deduper import Deduper

# Fetches the version of the package as defined in pyproject.toml
__version__ = pkg_resources.get_distribution("nlp_dedup").version
