"""DiT backbone and generator adapter."""

from .model import DiT
from .adapter import DiTBackboneAdapter, dit_xl_1, dit_xl_2, dit_xl_4

__all__ = ["DiT", "DiTBackboneAdapter", "dit_xl_1", "dit_xl_2", "dit_xl_4"]
