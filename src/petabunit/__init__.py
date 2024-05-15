"""petabunit - Python utilities for working with units with PEtab."""

from pathlib import Path

__version__ = "0.0.2"

program_name: str = "petabunit"
RESOURCES_DIR: Path = Path(__file__).parent / "resources"
ENUM_DIR: Path = Path(__file__).parent / "metadata"

CACHE_USE: bool = False
CACHE_PATH: Path = RESOURCES_DIR / "cache"
