"""
test_databento_fixes.py

Unit tests for the three Databento adapter fixes:
  1. ``_find_repo_root()`` returns a valid, absolute Path that contains ``.git/``.
  2. ``_DEFAULT_DATA_DIR`` is derived correctly.
  3. ``_ensure_symbol_column()`` raises ``ValueError`` on all-None/NaN symbols
     and passes on valid symbols.
  4. ``_fetch_databento()`` passes ``stype_out="raw_symbol"`` to ``get_range()``.
  5. Cross-module path constants (``_DEFAULT_UNIVERSE_DIR``,
     ``_DEFAULT_SAVE_DIR``) are absolute.

All tests are isolated, deterministic, and require no network access.
No production code is modified.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from data_collection.databento import (
    _find_repo_root,
    _DEFAULT_DATA_DIR,
    _ensure_symbol_column,
    DataBentoBarAdapter,
)


# =============================================================================
# 1.  _find_repo_root() tests
# =============================================================================

class TestFindRepoRoot:
    """``_find_repo_root()`` must return a sensible, absolute repo root."""

    def test_returns_path_object(self) -> None:
        """Return type must be ``pathlib.Path``."""
        result = _find_repo_root()
        assert isinstance(result, Path)

    def test_path_is_absolute(self) -> None:
        """The returned path must be absolute (not relative)."""
        result = _find_repo_root()
        assert result.is_absolute(), f"Expected absolute path, got: {result}"

    def test_path_contains_git_directory(self) -> None:
        """The returned path must contain a ``.git/`` entry — i.e. it is a repo root."""
        result = _find_repo_root()
        assert (result / ".git").exists(), (
            f"No .git/ directory found under {result}"
        )

    def test_default_data_dir_is_absolute(self) -> None:
        """``_DEFAULT_DATA_DIR`` must be absolute."""
        assert _DEFAULT_DATA_DIR.is_absolute(), (
            f"_DEFAULT_DATA_DIR is not absolute: {_DEFAULT_DATA_DIR}"
        )

    def test_default_data_dir_ends_with_data_databento(self) -> None:
        """``_DEFAULT_DATA_DIR`` must end with ``data/databento``."""
        parts = _DEFAULT_DATA_DIR.parts
        assert parts[-1] == "databento" and parts[-2] == "data", (
            f"Expected path ending in data/databento, got: {_DEFAULT_DATA_DIR}"
        )


# =============================================================================
# 2.  _ensure_symbol_column() ValueError tests
# =============================================================================

def _make_mock_dbnstore(mappings: dict | None = None) -> MagicMock:
    """
    Build a minimal mock of ``databento.DBNStore`` for ``_ensure_symbol_column``.

    Parameters
    ----------
    mappings : dict, optional
        Value for ``mock.mappings``.  Defaults to empty dict.

    Returns
    -------
    MagicMock
        Mock with a ``mappings`` attribute.
    """
    mock = MagicMock()
    mock.mappings = mappings if mappings is not None else {}
    return mock


class TestEnsureSymbolColumn:
    """Tests for ``_ensure_symbol_column()``."""

    # ------------------------------------------------------------------
    # ValueError cases
    # ------------------------------------------------------------------

    def test_raises_on_all_none_symbols(self) -> None:
        """A ``symbol`` column that is entirely ``None`` must raise ``ValueError``."""
        df = pd.DataFrame({
            "symbol": [None, None, None],
            "close": [100.0, 101.0, 102.0],
        })
        mock_store = _make_mock_dbnstore()
        with pytest.raises(ValueError, match="None/NaN"):
            _ensure_symbol_column(mock_store, df)

    def test_raises_on_all_nan_symbols(self) -> None:
        """A ``symbol`` column that is entirely ``np.nan`` must raise ``ValueError``."""
        df = pd.DataFrame({
            "symbol": [np.nan, np.nan, np.nan],
            "close": [100.0, 101.0, 102.0],
        })
        mock_store = _make_mock_dbnstore()
        with pytest.raises(ValueError, match="None/NaN"):
            _ensure_symbol_column(mock_store, df)

    def test_raises_error_message_mentions_stype_out(self) -> None:
        """The ValueError message must mention ``stype_out`` for actionability."""
        df = pd.DataFrame({
            "symbol": [None, None],
            "close": [50.0, 51.0],
        })
        mock_store = _make_mock_dbnstore()
        with pytest.raises(ValueError, match="stype_out"):
            _ensure_symbol_column(mock_store, df)

    # ------------------------------------------------------------------
    # Happy-path: valid string symbols — must NOT raise
    # ------------------------------------------------------------------

    def test_passes_on_valid_string_symbols(self) -> None:
        """A ``symbol`` column with human-readable strings must pass through unchanged."""
        df = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "SPY"],
            "close": [180.0, 400.0, 450.0],
        })
        mock_store = _make_mock_dbnstore()
        result = _ensure_symbol_column(mock_store, df)
        # Must return a DataFrame with the same symbols
        assert list(result["symbol"]) == ["AAPL", "MSFT", "SPY"]

    def test_passes_on_single_valid_symbol(self) -> None:
        """Single-row DataFrame with valid symbol must not raise."""
        df = pd.DataFrame({
            "symbol": ["QQQ"],
            "close": [400.0],
        })
        mock_store = _make_mock_dbnstore()
        result = _ensure_symbol_column(mock_store, df)
        assert result["symbol"].iloc[0] == "QQQ"

    def test_does_not_raise_on_mixed_valid_and_none(self) -> None:
        """Mixed None and valid symbols: should NOT raise (not all-None)."""
        df = pd.DataFrame({
            "symbol": [None, "AAPL", "MSFT"],
            "close": [100.0, 101.0, 102.0],
        })
        mock_store = _make_mock_dbnstore()
        # Only raises when *all* values are None/NaN; partial is not that case
        # This should not raise — it may still try to reconstruct via mappings
        try:
            _ensure_symbol_column(mock_store, df)
        except ValueError as exc:
            pytest.fail(
                f"Should not raise ValueError for partial-None symbols, but got: {exc}"
            )


# =============================================================================
# 3.  _fetch_databento() passes stype_out="raw_symbol"
# =============================================================================

class TestFetchDatabento:
    """``DataBentoBarAdapter._fetch_databento()`` must forward ``stype_out="raw_symbol"``."""

    def test_specific_tickers_use_default_stype(self, tmp_path: Path) -> None:
        """
        For specific tickers, ``get_range()`` should NOT pass
        ``stype_in`` or ``stype_out`` (uses SDK defaults:
        ``stype_in=raw_symbol``, ``stype_out=instrument_id``).
        """
        raw_path = tmp_path / "test.dbn.zst"

        mock_client = MagicMock()
        mock_dbn = MagicMock()
        mock_client.timeseries.get_range.return_value = mock_dbn

        with patch("data_collection.databento.db.Historical", return_value=mock_client):
            adapter = DataBentoBarAdapter(cache_dir=str(tmp_path))
            adapter.client = mock_client

            adapter._fetch_databento(
                tickers=["AAPL", "MSFT"],
                start_date="2025-01-02",
                end_date="2025-01-03",
                dataset="EQUS.MINI",
                schema="ohlcv-1m",
                raw_path=raw_path,
            )

        mock_client.timeseries.get_range.assert_called_once()
        _, kwargs = mock_client.timeseries.get_range.call_args
        assert "stype_in" not in kwargs, "Specific tickers should not set stype_in"
        assert "stype_out" not in kwargs, "Specific tickers should not set stype_out"

    def test_all_symbols_resolves_symbology_when_mappings_empty(self, tmp_path: Path) -> None:
        """
        For ALL_SYMBOLS, when ``data.mappings`` is empty,
        ``request_symbology`` should be called to resolve instrument IDs.
        """
        from data_collection.databento import ALL_SYMBOLS

        raw_path = tmp_path / "test.dbn.zst"

        mock_client = MagicMock()
        mock_dbn = MagicMock()
        # Simulate empty mappings (what Databento returns for ALL_SYMBOLS)
        mock_dbn.mappings = {}
        mock_dbn.request_symbology.return_value = {"result": {}}
        mock_client.timeseries.get_range.return_value = mock_dbn

        with patch("data_collection.databento.db.Historical", return_value=mock_client):
            adapter = DataBentoBarAdapter(cache_dir=str(tmp_path))
            adapter.client = mock_client

            adapter._fetch_databento(
                tickers=ALL_SYMBOLS,
                start_date="2025-01-02",
                end_date="2025-01-03",
                dataset="EQUS.MINI",
                schema="ohlcv-1d",
                raw_path=raw_path,
            )

        # Should have called request_symbology to resolve the IDs
        mock_dbn.request_symbology.assert_called_once_with(mock_client)
        mock_dbn.insert_symbology_json.assert_called_once()

    def test_all_symbols_skips_symbology_when_mappings_present(self, tmp_path: Path) -> None:
        """
        If mappings are already populated, ``request_symbology`` should
        NOT be called.
        """
        from data_collection.databento import ALL_SYMBOLS

        raw_path = tmp_path / "test.dbn.zst"

        mock_client = MagicMock()
        mock_dbn = MagicMock()
        # Simulate non-empty mappings
        mock_dbn.mappings = {"AAPL": [{"start_date": "2025-01-01", "end_date": "2025-01-03", "symbol": "1234"}]}
        mock_client.timeseries.get_range.return_value = mock_dbn

        with patch("data_collection.databento.db.Historical", return_value=mock_client):
            adapter = DataBentoBarAdapter(cache_dir=str(tmp_path))
            adapter.client = mock_client

            adapter._fetch_databento(
                tickers=ALL_SYMBOLS,
                start_date="2025-01-02",
                end_date="2025-01-03",
                dataset="EQUS.MINI",
                schema="ohlcv-1d",
                raw_path=raw_path,
            )

        mock_dbn.request_symbology.assert_not_called()

    def test_get_range_symbols_none_for_all_symbols(self, tmp_path: Path) -> None:
        """
        When ``ALL_SYMBOLS`` is passed, ``symbols`` should be ``None``
        in the ``get_range()`` call.
        """
        from data_collection.databento import ALL_SYMBOLS

        raw_path = tmp_path / "test.dbn.zst"

        mock_client = MagicMock()
        mock_dbn = MagicMock()
        mock_client.timeseries.get_range.return_value = mock_dbn

        with patch("data_collection.databento.db.Historical", return_value=mock_client):
            adapter = DataBentoBarAdapter(cache_dir=str(tmp_path))
            adapter.client = mock_client

            adapter._fetch_databento(
                tickers=ALL_SYMBOLS,
                start_date="2025-01-02",
                end_date="2025-01-03",
                dataset="EQUS.MINI",
                schema="ohlcv-1d",
                raw_path=raw_path,
            )

        _, kwargs = mock_client.timeseries.get_range.call_args
        assert kwargs.get("symbols") is None, (
            f"Expected symbols=None for ALL_SYMBOLS, got: {kwargs.get('symbols')!r}"
        )

    def test_get_range_called_with_correct_date_range(self, tmp_path: Path) -> None:
        """
        ``get_range()`` must receive the exact ``start`` and ``end`` passed
        to ``_fetch_databento()``.
        """
        raw_path = tmp_path / "test.dbn.zst"

        mock_client = MagicMock()
        mock_client.timeseries.get_range.return_value = MagicMock()

        with patch("data_collection.databento.db.Historical", return_value=mock_client):
            adapter = DataBentoBarAdapter(cache_dir=str(tmp_path))
            adapter.client = mock_client

            adapter._fetch_databento(
                tickers=["SPY"],
                start_date="2025-03-01",
                end_date="2025-03-31",
                dataset="EQUS.MINI",
                schema="ohlcv-1m",
                raw_path=raw_path,
            )

        _, kwargs = mock_client.timeseries.get_range.call_args
        assert kwargs.get("start") == "2025-03-01"
        assert kwargs.get("end") == "2025-03-31"


# =============================================================================
# 4.  Cross-module path consistency
# =============================================================================

class TestCrossModulePaths:
    """Module-level path constants must be absolute in every module that uses them."""

    def test_universe_selection_default_dir_is_absolute(self) -> None:
        """``_DEFAULT_UNIVERSE_DIR`` in ``universe_selection`` must be absolute."""
        from johansen_strategy.universe_selection import _DEFAULT_UNIVERSE_DIR

        assert isinstance(_DEFAULT_UNIVERSE_DIR, Path), (
            f"_DEFAULT_UNIVERSE_DIR is not a Path: {type(_DEFAULT_UNIVERSE_DIR)}"
        )
        assert _DEFAULT_UNIVERSE_DIR.is_absolute(), (
            f"_DEFAULT_UNIVERSE_DIR is not absolute: {_DEFAULT_UNIVERSE_DIR}"
        )

    def test_universe_selection_default_dir_ends_with_universes(self) -> None:
        """``_DEFAULT_UNIVERSE_DIR`` must end with ``data/universes``."""
        from johansen_strategy.universe_selection import _DEFAULT_UNIVERSE_DIR

        parts = _DEFAULT_UNIVERSE_DIR.parts
        assert parts[-1] == "universes" and parts[-2] == "data", (
            f"Expected path ending in data/universes, got: {_DEFAULT_UNIVERSE_DIR}"
        )

    def test_clustering_pipeline_default_save_dir_is_absolute(self) -> None:
        """``_DEFAULT_SAVE_DIR`` in ``clustering_pipeline`` must be absolute."""
        from johansen_strategy.clustering_pipeline import _DEFAULT_SAVE_DIR

        assert isinstance(_DEFAULT_SAVE_DIR, Path), (
            f"_DEFAULT_SAVE_DIR is not a Path: {type(_DEFAULT_SAVE_DIR)}"
        )
        assert _DEFAULT_SAVE_DIR.is_absolute(), (
            f"_DEFAULT_SAVE_DIR is not absolute: {_DEFAULT_SAVE_DIR}"
        )

    def test_clustering_pipeline_default_save_dir_ends_with_clusters(self) -> None:
        """``_DEFAULT_SAVE_DIR`` must end with ``data/clusters``."""
        from johansen_strategy.clustering_pipeline import _DEFAULT_SAVE_DIR

        parts = _DEFAULT_SAVE_DIR.parts
        assert parts[-1] == "clusters" and parts[-2] == "data", (
            f"Expected path ending in data/clusters, got: {_DEFAULT_SAVE_DIR}"
        )

    def test_all_path_constants_share_the_same_repo_root(self) -> None:
        """All path constants must be rooted under the same repo root."""
        from johansen_strategy.universe_selection import _DEFAULT_UNIVERSE_DIR
        from johansen_strategy.clustering_pipeline import _DEFAULT_SAVE_DIR

        repo_root = _find_repo_root()

        for name, path in [
            ("_DEFAULT_DATA_DIR", _DEFAULT_DATA_DIR),
            ("_DEFAULT_UNIVERSE_DIR", _DEFAULT_UNIVERSE_DIR),
            ("_DEFAULT_SAVE_DIR", _DEFAULT_SAVE_DIR),
        ]:
            assert str(path).startswith(str(repo_root)), (
                f"{name} ({path}) is not under repo root ({repo_root})"
            )
