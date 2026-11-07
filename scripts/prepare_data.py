#!/usr/bin/env python3
"""Thin wrapper for the ``prepare-data`` command.

Preferred invocation (after ``pip install -e .`` / ``poetry install``):

    poetry run fauxtography prepare-data --help
    poetry run fauxtography prepare-data --source-real R --source-fake F

This script simply forwards to the packaged CLI for convenience.
"""

from __future__ import annotations

import sys

from fauxtography.cli import main

if __name__ == "__main__":
    sys.exit(main(["prepare-data", *sys.argv[1:]]))