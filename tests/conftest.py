"""pytest configuration compartida."""

import pytest


@pytest.fixture
def sample_ohlcv():
    """DataFrame OHLCV de ejemplo para tests."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range("2026-04-01 13:30", periods=100, freq="1min", tz="UTC")
    np.random.seed(42)
    close = 150 + np.cumsum(np.random.randn(100) * 0.1)
    
    return pd.DataFrame({
        "ts": dates,
        "open": close + np.random.randn(100) * 0.05,
        "high": close + np.abs(np.random.randn(100) * 0.1),
        "low": close - np.abs(np.random.randn(100) * 0.1),
        "close": close,
        "volume": np.random.randint(10000, 50000, 100),
    }).set_index("ts")
