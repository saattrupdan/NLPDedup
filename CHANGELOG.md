# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this
project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [v0.1.2] - 2023-10-07
### Fixed
- Now allows Python versions above 3.10


## [v0.1.1] - 2022-12-20
### Fixed
- Import `Deduper` in `__init__.py`, allowing import of the class using
  `from nlp_dedup import Deduper`.


## [v0.1.0] - 2022-12-20
### Added
- Initial release, containing an implementation of the MinHash deduplication algorithm,
  both using the `Deduper` class and a CLI using the `deduper` command.
