"""Inverse molecular design modules."""

try:
    from .inverse import ScentDesigner
    _HAS_DESIGN = True
except ImportError:
    ScentDesigner = None
    _HAS_DESIGN = False

__all__ = []

if _HAS_DESIGN:
    __all__.append("ScentDesigner")