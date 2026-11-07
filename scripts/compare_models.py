#!/usr/bin/env python3
"""Thin wrapper for the ``compare`` command.

Preferred invocation (after ``pip install -e .`` / ``poetry install``):

    poetry run fauxtography compare --help
"""

from __future__ import annotations

import sys

from fauxtography.cli import main

if __name__ == "__main__":
    sys.exit(main(["compare", *sys.argv[1:]]))