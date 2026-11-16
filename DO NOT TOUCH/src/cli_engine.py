"""Headless CLI engine entry point for the greenfield universal alignment engine (v4.1).

Usage:
    python -m src.cli_engine --project "START HERE/_Projects/PaoPao - Grax" --align-only
"""

from __future__ import annotations

import sys
from typing import List, Optional

from src.cli import _find_sources, main as _cli_main

__all__ = ["_find_sources", "main"]


def main(argv: Optional[List[str]] = None) -> int:
    return _cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
