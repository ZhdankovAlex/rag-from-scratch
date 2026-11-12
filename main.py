"""
Entry point kept at the project root: `python main.py <command>`.

The CLI itself lives in rag_system/cli.py; `python -m rag_system` is equivalent.
"""
import sys

from rag_system.cli import main

if __name__ == "__main__":
    sys.exit(main())
