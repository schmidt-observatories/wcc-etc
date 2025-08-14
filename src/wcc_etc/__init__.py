import importlib.metadata
from pathlib import Path
__version__ = importlib.metadata.version(__package__ or "wcc_etc")

__all__ = [ "__version__"]

from .wcc_etc import WCCETC