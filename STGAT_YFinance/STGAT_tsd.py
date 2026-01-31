"""STGAT_tsd.py

Compatibility wrapper.

Some upstream scripts expect:
    from STGAT_tsd import GAT_TCN

The implementation lives in STGAT.py (this repo). This file re-exports it
so the project runs even if the original STGAT_tsd.py is missing.
"""

from STGAT import GAT_TCN

__all__ = ["GAT_TCN"]
