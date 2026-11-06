"""Load each backbone family only when its constructor is selected."""

from importlib import import_module

__all__ = [
    "pixeldit_b",
    "pixeldit_xl",
    "jit_b",
    "jit_l",
    "jit_h",
    "dit_xl_1",
    "dit_xl_2",
    "dit_xl_4",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    family = name.split("_", 1)[0]
    constructor = getattr(import_module(f".{family}", __name__), name)
    globals()[name] = constructor
    return constructor


def __dir__():
    return sorted(set(globals()) | set(__all__))
