#!/usr/bin/env python3
"""Thin wrapper for the ``eval-clip`` command.

Preferred invocation (after ``pip install -e .`` / ``poetry install``):

    poetry run fauxtography eval-clip --help
"""

from __future__ import annotations

import sys

from fauxtography.cli import main

if __name__ == "__main__":
    sys.exit(main(["eval-clip", *sys.argv[1:]]))