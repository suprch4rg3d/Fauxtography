#!/usr/bin/env python3
"""Thin wrapper for the ``extract-dinov2`` command.

Preferred invocation (after ``pip install -e .`` / ``poetry install``):

    poetry run fauxtography extract-dinov2 --help
"""

from __future__ import annotations

import sys

from fauxtography.cli import main

if __name__ == "__main__":
    sys.exit(main(["extract-dinov2", *sys.argv[1:]]))