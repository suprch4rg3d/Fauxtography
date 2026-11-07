#!/usr/bin/env python3
"""Thin wrapper for the ``train-linear-probe`` command.

Preferred invocation (after ``pip install -e .`` / ``poetry install``):

    poetry run fauxtography train-linear-probe --help
"""

from __future__ import annotations

import sys

from fauxtography.cli import main

if __name__ == "__main__":
    sys.exit(main(["train-linear-probe", *sys.argv[1:]]))