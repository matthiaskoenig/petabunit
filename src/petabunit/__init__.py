"""petabunit - Python utilities for working with units with PEtab."""

from pathlib import Path

__version__ = "0.0.2"

program_name: str = "petabunit"
BASE_DIR: Path = Path(__file__).parent.parent.parent
EXAMPLES_DIR: Path = BASE_DIR / "examples"
