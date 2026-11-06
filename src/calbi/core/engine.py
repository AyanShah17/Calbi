"""Calbi Engine facade re-export for backwards compatibility."""

from __future__ import annotations

from calbi.facade import Calbi, set_delegate_handler

__all__ = [
    "Calbi",
    "set_delegate_handler",
]
