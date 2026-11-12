"""Allow `python -m rag_system <command>`."""
import sys

from .cli import main

sys.exit(main())
