"""gen_data_tsd.py

Compatibility wrapper.

Original scripts import:
    from gen_data_tsd import gen_GNN_data

This module forwards to the Yahoo Finance-backed implementation.
"""

from gen_data_yf_tsd import gen_GNN_data, YFConfig

__all__ = ["gen_GNN_data", "YFConfig"]
