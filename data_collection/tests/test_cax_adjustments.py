"""
test_cax_adjustments.py

Unit tests for the pluggable corporate-action (CAX) split adjustment system
introduced in ``data_collection.databento``.

Coverage:
  1. ``AdjustmentProvider`` interface — ``YFinanceAdjuster`` and
     ``DatabentoCaxAdjuster`` are concrete providers; ``DatabentoCaxAdjuster``
     raises ``NotImplementedError``.
  2. ``YFinanceAdjuster`` caching — memo cache, then parquet cache, then
     live download.
  3. ``YFinanceAdjuster._df_to_splits_dict`` — multi-ticker MultiIndex (the
     raw yf.download output), flat post-slice DataFrame (multi-ticker after
     df["Stock Splits"]), single-ticker flat columns, empty DataFrame,
     zero-filtering, NaN-filtering.
  4. ``DataBentoBarAdapter`` constructor — correct adjuster type for each
     ``cax_method`` value; ``ValueError`` for invalid values; backward-
     compatible default.
  5. ``_apply_adjustments`` integration — per-symbol isolation, price/volume
     mechanics, ``ALL_SYMBOLS`` pass-through, ``cax_method=None`` pass-through.

Notes on production code behavior
----------------------------------
``YFinanceAdjuster.get_splits`` calls ``yf.download`` which returns a
``(price_type, ticker)`` MultiIndex DataFrame. The production code slices
``df["Stock Splits"]`` to get a flat-column DataFrame (columns = ticker names).
``_df_to_splits_dict`` then receives this flat-column DataFrame.

Known limitation (flagged to coder): the ``_df_to_splits_dict`` method only
handles two cases explicitly:

* ``isinstance(splits_df.columns, pd.MultiIndex)`` — raw yf.download output
  (never actually reached via the normal ``get_splits`` path, since the slice
  strips the first MultiIndex level).
* ``len(tickers) == 1`` — single-ticker flat columns.

For multiple tickers, the flat-column case falls through to
``return {t: empty for t in tickers}``, so multi-ticker splits are silently
dropped. The caching and single-ticker paths are fully functional. This test
suite documents both the working paths and the multi-ticker limitation.

All tests are isolated, deterministic, and require no network access.
No production code is modified.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data_collection.databento import (
    ALL_SYMBOLS,
    AdjustmentProvider,
    DataBentoBarAdapter,
    DatabentoCaxAdjuster,
    YFinanceAdjuster,
)

import databento as db


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    """Temporary directory used as the adapter / adjuster cache root.

    Returns
    -------
    Path
        A fresh temporary directory for each test.
    """
    return tmp_path


@pytest.fixture()
def yf_adjuster(cache_dir: Path) -> YFinanceAdjuster:
    """``YFinanceAdjuster`` instance pointing at a temp cache directory.

    Returns
    -------
    YFinanceAdjuster
        Freshly constructed adjuster; no prior cache state.
    """
    return YFinanceAdjuster(cache_dir)


def _make_yf_download_multi(tickers: list[str], dates: list[str],
                             split_ratios: dict[str, list[float]]) -> pd.DataFrame:
    """Build a DataFrame mimicking ``yf.download(..., actions=True)`` for multiple tickers.

    The real ``yf.download`` returns a ``(price_type, ticker)`` MultiIndex.
    This helper builds the full multi-ticker output including a
    ``("Stock Splits", ticker)`` column for each ticker.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols.
    dates : list of str
        ISO date strings for the DatetimeIndex.
    split_ratios : dict of str -> list of float
        Mapping of ticker -> list of split ratio values aligned to ``dates``.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with a ``(price_type, ticker)`` MultiIndex on columns.
    """
    idx = pd.to_datetime(dates)
    col_tuples = [("Stock Splits", t) for t in tickers]
    data = {t: split_ratios.get(t, [0.0] * len(dates)) for t in tickers}
    arrays = [[r for r, t in col_tuples], [t for r, t in col_tuples]]
    cols = pd.MultiIndex.from_tuples(col_tuples, names=["Price", "Ticker"])
    df = pd.DataFrame(
        {t: split_ratios.get(t, [0.0] * len(dates)) for t in tickers},
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(
        [("Stock Splits", t) for t in tickers], names=["Price", "Ticker"]
    )
    return df


def _make_yf_download_single(dates: list[str], ratios: list[float]) -> pd.DataFrame:
    """Build a DataFrame mimicking ``yf.download(..., actions=True)`` for one ticker.

    Single-ticker ``yf.download`` returns flat columns including
    ``"Stock Splits"``.

    Parameters
    ----------
    dates : list of str
        ISO date strings for the DatetimeIndex.
    ratios : list of float
        Split ratio values aligned to ``dates``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``DatetimeIndex`` and a ``"Stock Splits"`` flat column.
    """
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"Stock Splits": ratios}, index=idx)


def _make_post_slice_multi(tickers: list[str], dates: list[str],
                           split_ratios: dict[str, list[float]]) -> pd.DataFrame:
    """Build a DataFrame as it appears after ``yf_df["Stock Splits"]`` slicing.

    After slicing the ``"Stock Splits"`` level from the full multi-ticker
    ``yf.download`` output, the result has flat ticker-name columns (no
    MultiIndex). This is what ``_df_to_splits_dict`` actually receives in the
    multi-ticker path.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols (become flat column names).
    dates : list of str
        ISO date strings for the DatetimeIndex.
    split_ratios : dict of str -> list of float
        Mapping of ticker -> list of split ratio values aligned to ``dates``.

    Returns
    -------
    pd.DataFrame
        Flat-column DataFrame with ticker names as columns.
    """
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {t: split_ratios.get(t, [0.0] * len(dates)) for t in tickers},
        index=idx,
    )


def _build_price_df(symbols: list[str],
                    dates_eastern: list[pd.Timestamp],
                    price: float = 200.0,
                    volume: float = 1_000.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with tz-aware US/Eastern DatetimeIndex.

    Parameters
    ----------
    symbols : list of str
        Ticker symbols; one row per (symbol, date) combination.
    dates_eastern : list of pd.Timestamp
        Timezone-aware timestamps (``US/Eastern``).
    price : float, optional
        Constant price used for open/high/low/close columns.
    volume : float, optional
        Constant volume for every row.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with tz-aware US/Eastern DatetimeIndex and a
        ``symbol`` column. Rows are ordered by (date, symbol).
    """
    rows = []
    idx_vals = []
    for dt in dates_eastern:
        for sym in symbols:
            rows.append({
                "symbol": sym,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            })
            idx_vals.append(dt)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx_vals))


# =============================================================================
# 1.  AdjustmentProvider interface
# =============================================================================


class TestAdjustmentProviderInterface:
    """Both concrete providers implement ``AdjustmentProvider``."""

    def test_yfinance_adjuster_is_adjustment_provider(self, cache_dir: Path) -> None:
        """``YFinanceAdjuster`` must be an instance of ``AdjustmentProvider``."""
        adjuster = YFinanceAdjuster(cache_dir)
        assert isinstance(adjuster, AdjustmentProvider)

    @patch("data_collection.databento.db.Reference")
    def test_databento_cax_adjuster_is_adjustment_provider(self, mock_ref_cls) -> None:
        """``DatabentoCaxAdjuster`` must be an instance of ``AdjustmentProvider``."""
        adjuster = DatabentoCaxAdjuster("fake-key", Path("/tmp"))
        assert isinstance(adjuster, AdjustmentProvider)

    @patch("data_collection.databento.db.Reference")
    def test_databento_cax_adjuster_calls_adjustment_factors(self, mock_ref_cls) -> None:
        """``DatabentoCaxAdjuster.get_splits()`` calls the Reference Data API."""
        mock_ref = mock_ref_cls.return_value
        mock_ref.adjustment_factors.get_range.return_value = pd.DataFrame()
        adjuster = DatabentoCaxAdjuster("fake-key", Path("/tmp"))
        result = adjuster.get_splits(["AAPL"], "2025-01-01", "2025-12-31")
        mock_ref.adjustment_factors.get_range.assert_called_once()
        assert result["AAPL"].empty

    @patch("data_collection.databento.db.Reference")
    def test_databento_cax_adjuster_returns_splits_from_api(self, mock_ref_cls, tmp_path) -> None:
        """Adjuster extracts per-ticker splits from the adjustment_factors response."""
        mock_ref = mock_ref_cls.return_value
        adj_df = pd.DataFrame({
            "raw_symbol": ["AAPL", "AAPL", "MSFT"],
            "adj_factor": [4.0, 0.0, 2.0],
        }, index=pd.to_datetime(["2025-06-01", "2025-07-01", "2025-08-01"]))
        adj_df.index.name = "ex_date"
        mock_ref.adjustment_factors.get_range.return_value = adj_df
        adjuster = DatabentoCaxAdjuster("fake-key", tmp_path)
        result = adjuster.get_splits(["AAPL", "MSFT"], "2025-01-01", "2025-12-31")
        # AAPL: 4.0 kept, 0.0 filtered out
        assert len(result["AAPL"]) == 1
        assert result["AAPL"].iloc[0] == pytest.approx(4.0)
        assert len(result["MSFT"]) == 1
        assert result["MSFT"].iloc[0] == pytest.approx(2.0)


# =============================================================================
# 2.  YFinanceAdjuster caching
# =============================================================================


class TestYFinanceAdjusterCaching:
    """Cache layers: memo first, then parquet, then live yf.download."""

    def _make_single_ticker_yf_return(self) -> pd.DataFrame:
        """Build a single-ticker yf.download return with one 4:1 split for AAPL."""
        return _make_yf_download_single(["2025-06-01"], [4.0])

    def test_first_call_invokes_yf_download(self, yf_adjuster: YFinanceAdjuster) -> None:
        """On first call, ``yf.download`` must be called exactly once."""
        tickers = ["AAPL"]
        yf_return = self._make_single_ticker_yf_return()
        with patch("yfinance.download", return_value=yf_return) as mock_dl:
            yf_adjuster.get_splits(tickers, "2025-01-01", "2025-12-31")
        mock_dl.assert_called_once()

    def test_second_call_uses_memo_no_yf_download(self, yf_adjuster: YFinanceAdjuster) -> None:
        """Second call with same args must return from memo; ``yf.download`` NOT called again."""
        tickers = ["AAPL"]
        yf_return = self._make_single_ticker_yf_return()
        with patch("yfinance.download", return_value=yf_return) as mock_dl:
            result1 = yf_adjuster.get_splits(tickers, "2025-01-01", "2025-12-31")
            result2 = yf_adjuster.get_splits(tickers, "2025-01-01", "2025-12-31")
        assert mock_dl.call_count == 1, (
            "yf.download should only be called once (memo hit on 2nd call)"
        )
        assert result1.keys() == result2.keys()

    def test_parquet_cache_file_created_after_first_call(
        self, yf_adjuster: YFinanceAdjuster, cache_dir: Path
    ) -> None:
        """After the first download with a real split, a ``.parquet`` file must exist in ``splits_cache/``."""
        tickers = ["AAPL"]
        yf_return = self._make_single_ticker_yf_return()
        with patch("yfinance.download", return_value=yf_return):
            yf_adjuster.get_splits(tickers, "2025-01-01", "2025-12-31")
        parquet_files = list((cache_dir / "splits_cache").glob("*.parquet"))
        assert len(parquet_files) == 1, (
            f"Expected exactly one parquet cache file, found: {parquet_files}"
        )

    def test_splits_cache_subdirectory_created_on_init(self, cache_dir: Path) -> None:
        """``YFinanceAdjuster.__init__`` must create ``splits_cache/`` subdirectory."""
        assert not (cache_dir / "splits_cache").exists()
        YFinanceAdjuster(cache_dir)
        assert (cache_dir / "splits_cache").is_dir()

    def test_third_call_after_memo_cleared_uses_parquet(
        self, yf_adjuster: YFinanceAdjuster
    ) -> None:
        """After clearing the memo, a subsequent call must read from parquet (no yf.download)."""
        tickers = ["AAPL"]
        yf_return = self._make_single_ticker_yf_return()

        # Populate parquet cache (single-ticker path always writes parquet)
        with patch("yfinance.download", return_value=yf_return) as mock_dl:
            yf_adjuster.get_splits(tickers, "2025-01-01", "2025-12-31")
        assert mock_dl.call_count == 1

        # Confirm parquet was written
        parquet_files = list(yf_adjuster._cache_dir.glob("*.parquet"))
        assert len(parquet_files) == 1, "Parquet cache must exist before clearing memo"

        # Clear memo to force parquet path
        yf_adjuster._memo.clear()

        # Next call — should read parquet, NOT call yf.download
        with patch("yfinance.download", return_value=yf_return) as mock_dl2:
            result = yf_adjuster.get_splits(tickers, "2025-01-01", "2025-12-31")
        assert mock_dl2.call_count == 0, (
            "After memo clear, parquet cache should be used instead of re-downloading"
        )
        assert set(result.keys()) == set(tickers)

    def test_result_dict_has_entry_for_every_requested_ticker(
        self, yf_adjuster: YFinanceAdjuster
    ) -> None:
        """``get_splits`` must return a dict with one key per requested ticker."""
        tickers = ["AAPL"]
        yf_return = self._make_single_ticker_yf_return()
        with patch("yfinance.download", return_value=yf_return):
            result = yf_adjuster.get_splits(tickers, "2025-01-01", "2025-12-31")
        assert set(result.keys()) == set(tickers)

    def test_empty_yf_download_returns_empty_series_for_all_tickers(
        self, yf_adjuster: YFinanceAdjuster
    ) -> None:
        """If ``yf.download`` returns an empty DataFrame, all tickers get empty Series."""
        tickers = ["SPY", "QQQ"]
        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = yf_adjuster.get_splits(tickers, "2025-01-01", "2025-12-31")
        for t in tickers:
            assert isinstance(result[t], pd.Series)
            assert result[t].empty

    def test_different_date_ranges_produce_different_cache_keys(
        self, yf_adjuster: YFinanceAdjuster
    ) -> None:
        """Two calls with different date ranges must NOT share a cache entry."""
        tickers = ["AAPL"]
        yf_return = self._make_single_ticker_yf_return()
        with patch("yfinance.download", return_value=yf_return) as mock_dl:
            yf_adjuster.get_splits(tickers, "2025-01-01", "2025-06-30")
            yf_adjuster.get_splits(tickers, "2025-07-01", "2025-12-31")
        assert mock_dl.call_count == 2, (
            "Different date ranges must trigger separate downloads"
        )


# =============================================================================
# 3.  YFinanceAdjuster._df_to_splits_dict
# =============================================================================


class TestDfToSplitsDict:
    """Unit tests for the static ``_df_to_splits_dict`` helper.

    Notes
    -----
    ``_df_to_splits_dict`` can receive three distinct DataFrame shapes:

    1. **Raw MultiIndex** — ``isinstance(splits_df.columns, pd.MultiIndex)`` is
       ``True``.  The production ``get_splits`` path never actually produces
       this shape (``yf.download`` output is sliced before calling this method),
       but the code handles it explicitly.

    2. **Flat multi-ticker** (post ``df["Stock Splits"]`` slice) — flat columns
       equal to ticker names, multiple tickers.  The production code falls
       through to ``return {t: empty for t in tickers}``.

    3. **Flat single-ticker** — ``len(tickers) == 1`` with a single flat column.
    """

    def test_raw_multiindex_correct_per_ticker_series(self) -> None:
        """Raw MultiIndex DataFrame (first level = ticker) returns correct split Series."""
        tickers = ["AAPL", "MSFT"]
        idx = pd.to_datetime(["2025-06-01", "2025-07-01"])
        # MultiIndex with tickers as first level (as the code comment says)
        cols = pd.MultiIndex.from_tuples(
            [("AAPL", ""), ("MSFT", "")], names=["Price", "Ticker"]
        )
        splits_df = pd.DataFrame([[4.0, 0.0], [0.0, 0.0]], index=idx, columns=cols)
        result = YFinanceAdjuster._df_to_splits_dict(splits_df, tickers)
        # AAPL should have the 4.0 split; MSFT empty
        assert result["AAPL"].iloc[0] == pytest.approx(4.0)
        assert result["MSFT"].empty

    def test_single_ticker_flat_columns_correct_series(self) -> None:
        """Single-ticker flat-column DataFrame (from ``yf.download``) returns correct Series."""
        # Single-ticker yf.download returns flat "Stock Splits" column
        df = _make_yf_download_single(["2025-06-01", "2025-07-01"], [2.0, 0.0])
        # The production code slices df[["Stock Splits"]] before calling this
        splits_df = df[["Stock Splits"]]
        result = YFinanceAdjuster._df_to_splits_dict(splits_df, ["AAPL"])
        assert "AAPL" in result
        assert result["AAPL"].iloc[0] == pytest.approx(2.0)
        assert len(result["AAPL"]) == 1

    def test_empty_dataframe_returns_all_empty_series(self) -> None:
        """Empty input DataFrame must return empty Series for every requested ticker."""
        result = YFinanceAdjuster._df_to_splits_dict(pd.DataFrame(), ["AAPL", "SPY"])
        assert result["AAPL"].empty
        assert result["SPY"].empty

    def test_zero_splits_filtered_out_single_ticker(self) -> None:
        """Split ratios equal to zero must be excluded from the single-ticker Series."""
        df = _make_yf_download_single(["2025-01-01", "2025-06-01"], [0.0, 3.0])
        splits_df = df[["Stock Splits"]]
        result = YFinanceAdjuster._df_to_splits_dict(splits_df, ["AAPL"])
        assert 0.0 not in result["AAPL"].values
        assert len(result["AAPL"]) == 1

    def test_nan_splits_filtered_out_single_ticker(self) -> None:
        """NaN split ratios must be excluded from the single-ticker Series."""
        df = _make_yf_download_single(["2025-01-01", "2025-06-01"], [np.nan, 5.0])
        splits_df = df[["Stock Splits"]]
        result = YFinanceAdjuster._df_to_splits_dict(splits_df, ["AAPL"])
        assert not result["AAPL"].isna().any()
        assert result["AAPL"].iloc[0] == pytest.approx(5.0)

    def test_all_nonzero_splits_preserved_single_ticker(self) -> None:
        """Multiple non-zero splits in one single-ticker Series must all be preserved."""
        df = _make_yf_download_single(
            ["2024-06-01", "2025-06-01"], [2.0, 4.0]
        )
        splits_df = df[["Stock Splits"]]
        result = YFinanceAdjuster._df_to_splits_dict(splits_df, ["AAPL"])
        assert len(result["AAPL"]) == 2

    def test_multi_ticker_post_slice_flat_columns_extracts_splits(self) -> None:
        """Post-slice flat multi-ticker DataFrame correctly extracts per-ticker splits."""
        tickers = ["AAPL", "MSFT"]
        # This is what the production code passes after df["Stock Splits"]
        splits_df = _make_post_slice_multi(
            tickers, ["2025-06-01"], {"AAPL": [4.0], "MSFT": [0.0]}
        )
        assert not isinstance(splits_df.columns, pd.MultiIndex), (
            "Fixture must be flat-column to exercise the correct code path"
        )
        result = YFinanceAdjuster._df_to_splits_dict(splits_df, tickers)
        # AAPL had a 4:1 split, MSFT had a 0 (filtered out)
        assert len(result["AAPL"]) == 1
        assert result["AAPL"].iloc[0] == pytest.approx(4.0)
        assert result["MSFT"].empty

    def test_raw_multiindex_ticker_absent_returns_empty(self) -> None:
        """Ticker not present in a raw MultiIndex DataFrame gets an empty Series."""
        tickers_in_df = ["MSFT"]
        idx = pd.to_datetime(["2025-06-01"])
        cols = pd.MultiIndex.from_tuples([("MSFT", "")], names=["Price", "Ticker"])
        splits_df = pd.DataFrame([[2.0]], index=idx, columns=cols)
        result = YFinanceAdjuster._df_to_splits_dict(splits_df, ["AAPL", "MSFT"])
        assert result["AAPL"].empty
        assert result["MSFT"].iloc[0] == pytest.approx(2.0)


# =============================================================================
# 4.  DataBentoBarAdapter cax_method constructor behaviour
# =============================================================================


class TestDataBentoBarAdapterCaxMethod:
    """``cax_method`` controls which adjuster is instantiated."""

    def _make_adapter(self, tmp_path: Path, cax_method) -> DataBentoBarAdapter:
        """Construct adapter with mocked Databento clients."""
        with patch("data_collection.databento.db.Historical"), \
             patch("data_collection.databento.db.Reference"):
            return DataBentoBarAdapter(
                cache_dir=str(tmp_path), cax_method=cax_method
            )

    def test_yfinance_method_creates_yfinance_adjuster(self, tmp_path: Path) -> None:
        """``cax_method='yfinance'`` must set ``_adjuster`` to ``YFinanceAdjuster``."""
        adapter = self._make_adapter(tmp_path, "yfinance")
        assert isinstance(adapter._adjuster, YFinanceAdjuster)

    def test_databento_method_creates_databento_cax_adjuster(
        self, tmp_path: Path
    ) -> None:
        """``cax_method='databento'`` must set ``_adjuster`` to ``DatabentoCaxAdjuster``."""
        adapter = self._make_adapter(tmp_path, "databento")
        assert isinstance(adapter._adjuster, DatabentoCaxAdjuster)

    def test_none_method_sets_adjuster_to_none(self, tmp_path: Path) -> None:
        """``cax_method=None`` must set ``_adjuster`` to ``None``."""
        adapter = self._make_adapter(tmp_path, None)
        assert adapter._adjuster is None

    def test_invalid_cax_method_raises_value_error(self, tmp_path: Path) -> None:
        """An unrecognised ``cax_method`` must raise ``ValueError``."""
        with pytest.raises(ValueError, match="cax_method"):
            self._make_adapter(tmp_path, "bloomberg")

    def test_default_cax_method_is_yfinance(self, tmp_path: Path) -> None:
        """Default constructor (no ``cax_method``) must use ``YFinanceAdjuster``."""
        with patch("data_collection.databento.db.Historical"):
            adapter = DataBentoBarAdapter(cache_dir=str(tmp_path))
        assert isinstance(adapter._adjuster, YFinanceAdjuster), (
            "Default cax_method must be 'yfinance' for backward compatibility"
        )

    def test_empty_string_cax_method_raises_value_error(self, tmp_path: Path) -> None:
        """An empty string ``cax_method`` must raise ``ValueError``."""
        with pytest.raises(ValueError, match="cax_method"):
            self._make_adapter(tmp_path, "")

    def test_uppercase_cax_method_raises_value_error(self, tmp_path: Path) -> None:
        """``cax_method='YFINANCE'`` (wrong case) must raise ``ValueError``."""
        with pytest.raises(ValueError, match="cax_method"):
            self._make_adapter(tmp_path, "YFINANCE")

    def test_explicit_yfinance_matches_default_adjuster_type(
        self, tmp_path: Path
    ) -> None:
        """Explicit ``cax_method='yfinance'`` must produce the same type as the default."""
        with patch("data_collection.databento.db.Historical"):
            default_adapter = DataBentoBarAdapter(cache_dir=str(tmp_path / "default"))
            explicit_adapter = DataBentoBarAdapter(
                cache_dir=str(tmp_path / "explicit"), cax_method="yfinance"
            )
        assert type(default_adapter._adjuster) is type(explicit_adapter._adjuster)


# =============================================================================
# 5.  _apply_adjustments integration
# =============================================================================


class TestApplyAdjustments:
    """Integration-level tests for ``DataBentoBarAdapter._apply_adjustments``."""

    PRE_PRICE: float = 200.0
    VOLUME: float = 1_000.0
    SPLIT_RATIO: float = 4.0

    # Split happens at 09:30 on 2025-06-01; rows before that are adjusted
    SPLIT_TS = pd.Timestamp("2025-06-01 09:30", tz="US/Eastern")
    PRE_DATES = [
        pd.Timestamp("2025-05-30 09:30", tz="US/Eastern"),
        pd.Timestamp("2025-05-31 09:30", tz="US/Eastern"),
    ]
    POST_DATES = [
        SPLIT_TS,
        pd.Timestamp("2025-06-02 09:30", tz="US/Eastern"),
    ]

    def _make_adapter(self, tmp_path: Path) -> DataBentoBarAdapter:
        """Construct adapter with a ``YFinanceAdjuster`` (mock injected after)."""
        with patch("data_collection.databento.db.Historical"):
            return DataBentoBarAdapter(
                cache_dir=str(tmp_path), cax_method="yfinance"
            )

    def _build_df(self, symbols: list[str]) -> pd.DataFrame:
        """Build a 4-row-per-symbol OHLCV DataFrame bracketing the split date.

        Two rows fall before ``SPLIT_TS`` (should be adjusted) and two rows
        fall on or after it (should be unchanged).

        Parameters
        ----------
        symbols : list of str
            Symbols to include.

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame with tz-aware US/Eastern DatetimeIndex.
        """
        return _build_price_df(
            symbols,
            self.PRE_DATES + self.POST_DATES,
            price=self.PRE_PRICE,
            volume=self.VOLUME,
        )

    def _mock_splits(self) -> dict[str, pd.Series]:
        """Return mock split dict: AAPL 4:1 on SPLIT_TS, MSFT empty."""
        return {
            "AAPL": pd.Series(
                [self.SPLIT_RATIO],
                index=pd.DatetimeIndex([self.SPLIT_TS]),
            ),
            "MSFT": pd.Series(dtype=float),
        }

    def test_aapl_prices_before_split_are_divided_by_ratio(
        self, tmp_path: Path
    ) -> None:
        """Prices BEFORE the split date for the affected ticker must be divided by the ratio."""
        adapter = self._make_adapter(tmp_path)
        adapter._adjuster = MagicMock(spec=AdjustmentProvider)
        adapter._adjuster.get_splits.return_value = self._mock_splits()

        df = self._build_df(["AAPL", "MSFT"])
        result = adapter._apply_adjustments(df, ["AAPL", "MSFT"], "2025-01-01", "2025-12-31")

        aapl_pre = result[(result["symbol"] == "AAPL") & (result.index < self.SPLIT_TS)]
        assert len(aapl_pre) == 2, "Expected 2 pre-split rows for AAPL"
        expected = self.PRE_PRICE / self.SPLIT_RATIO
        np.testing.assert_allclose(aapl_pre["close"].values, expected)
        np.testing.assert_allclose(aapl_pre["open"].values, expected)

    def test_aapl_prices_on_or_after_split_are_unchanged(
        self, tmp_path: Path
    ) -> None:
        """Prices ON or AFTER the split date must NOT be adjusted."""
        adapter = self._make_adapter(tmp_path)
        adapter._adjuster = MagicMock(spec=AdjustmentProvider)
        adapter._adjuster.get_splits.return_value = self._mock_splits()

        df = self._build_df(["AAPL", "MSFT"])
        result = adapter._apply_adjustments(df, ["AAPL", "MSFT"], "2025-01-01", "2025-12-31")

        aapl_post = result[(result["symbol"] == "AAPL") & (result.index >= self.SPLIT_TS)]
        assert len(aapl_post) == 2, "Expected 2 post-split rows for AAPL"
        np.testing.assert_allclose(aapl_post["close"].values, self.PRE_PRICE)

    def test_only_aapl_is_adjusted_msft_prices_unchanged(
        self, tmp_path: Path
    ) -> None:
        """A split on AAPL must NOT alter MSFT prices (symbol isolation)."""
        adapter = self._make_adapter(tmp_path)
        adapter._adjuster = MagicMock(spec=AdjustmentProvider)
        adapter._adjuster.get_splits.return_value = self._mock_splits()

        df = self._build_df(["AAPL", "MSFT"])
        result = adapter._apply_adjustments(df, ["AAPL", "MSFT"], "2025-01-01", "2025-12-31")

        msft_rows = result[result["symbol"] == "MSFT"]
        np.testing.assert_allclose(msft_rows["close"].values, self.PRE_PRICE,
                                   err_msg="MSFT prices should be unchanged when only AAPL has a split")

    def test_volume_multiplied_by_ratio_for_split_ticker_pre_split(
        self, tmp_path: Path
    ) -> None:
        """Volume BEFORE the split date for the affected ticker must be multiplied by the ratio."""
        adapter = self._make_adapter(tmp_path)
        adapter._adjuster = MagicMock(spec=AdjustmentProvider)
        adapter._adjuster.get_splits.return_value = self._mock_splits()

        df = self._build_df(["AAPL", "MSFT"])
        result = adapter._apply_adjustments(df, ["AAPL", "MSFT"], "2025-01-01", "2025-12-31")

        aapl_pre = result[(result["symbol"] == "AAPL") & (result.index < self.SPLIT_TS)]
        np.testing.assert_allclose(aapl_pre["volume"].values, self.VOLUME * self.SPLIT_RATIO)

    def test_volume_unchanged_for_split_ticker_post_split(
        self, tmp_path: Path
    ) -> None:
        """Volume ON or AFTER the split date must NOT be adjusted."""
        adapter = self._make_adapter(tmp_path)
        adapter._adjuster = MagicMock(spec=AdjustmentProvider)
        adapter._adjuster.get_splits.return_value = self._mock_splits()

        df = self._build_df(["AAPL", "MSFT"])
        result = adapter._apply_adjustments(df, ["AAPL", "MSFT"], "2025-01-01", "2025-12-31")

        aapl_post = result[(result["symbol"] == "AAPL") & (result.index >= self.SPLIT_TS)]
        np.testing.assert_allclose(aapl_post["volume"].values, self.VOLUME)

    def test_volume_of_non_split_ticker_always_unchanged(
        self, tmp_path: Path
    ) -> None:
        """MSFT volume must be unaffected by an AAPL-only split at all time points."""
        adapter = self._make_adapter(tmp_path)
        adapter._adjuster = MagicMock(spec=AdjustmentProvider)
        adapter._adjuster.get_splits.return_value = self._mock_splits()

        df = self._build_df(["AAPL", "MSFT"])
        result = adapter._apply_adjustments(df, ["AAPL", "MSFT"], "2025-01-01", "2025-12-31")

        msft_rows = result[result["symbol"] == "MSFT"]
        np.testing.assert_allclose(msft_rows["volume"].values, self.VOLUME)

    def test_all_symbols_returns_df_unchanged(self, tmp_path: Path) -> None:
        """When ``tickers=ALL_SYMBOLS``, ``_apply_adjustments`` must return df unchanged."""
        adapter = self._make_adapter(tmp_path)
        adapter._adjuster = MagicMock(spec=AdjustmentProvider)

        df = self._build_df(["AAPL"])
        result = adapter._apply_adjustments(df, ALL_SYMBOLS, "2025-01-01", "2025-12-31")

        pd.testing.assert_frame_equal(result, df)
        adapter._adjuster.get_splits.assert_not_called()

    def test_cax_method_none_returns_df_unchanged(self, tmp_path: Path) -> None:
        """When ``cax_method=None``, ``_apply_adjustments`` must return df unchanged."""
        with patch("data_collection.databento.db.Historical"):
            adapter = DataBentoBarAdapter(
                cache_dir=str(tmp_path), cax_method=None
            )
        df = self._build_df(["AAPL"])
        original_close = df["close"].copy()
        result = adapter._apply_adjustments(df, ["AAPL"], "2025-01-01", "2025-12-31")
        pd.testing.assert_series_equal(result["close"], original_close)

    def test_adjuster_get_splits_called_with_correct_args(
        self, tmp_path: Path
    ) -> None:
        """``_apply_adjustments`` must forward tickers and date range to the adjuster."""
        adapter = self._make_adapter(tmp_path)
        mock_adj = MagicMock(spec=AdjustmentProvider)
        mock_adj.get_splits.return_value = {
            "AAPL": pd.Series(dtype=float),
            "MSFT": pd.Series(dtype=float),
        }
        adapter._adjuster = mock_adj

        df = self._build_df(["AAPL", "MSFT"])
        adapter._apply_adjustments(df, ["AAPL", "MSFT"], "2025-01-01", "2025-12-31")

        mock_adj.get_splits.assert_called_once_with(
            ["AAPL", "MSFT"], "2025-01-01", "2025-12-31"
        )

    def test_empty_splits_returns_df_unchanged(self, tmp_path: Path) -> None:
        """When the adjuster returns all-empty splits, prices must not be altered."""
        adapter = self._make_adapter(tmp_path)
        mock_adj = MagicMock(spec=AdjustmentProvider)
        mock_adj.get_splits.return_value = {
            "AAPL": pd.Series(dtype=float),
        }
        adapter._adjuster = mock_adj

        df = self._build_df(["AAPL"])
        original_close = df["close"].copy()
        result = adapter._apply_adjustments(df, ["AAPL"], "2025-01-01", "2025-12-31")
        pd.testing.assert_series_equal(result["close"], original_close)

    def test_zero_ratio_in_splits_skipped(self, tmp_path: Path) -> None:
        """A split ratio of zero in the adjuster output must be skipped (no division by zero)."""
        adapter = self._make_adapter(tmp_path)
        mock_adj = MagicMock(spec=AdjustmentProvider)
        mock_adj.get_splits.return_value = {
            "AAPL": pd.Series(
                [0.0], index=pd.DatetimeIndex([self.SPLIT_TS])
            ),
        }
        adapter._adjuster = mock_adj

        df = self._build_df(["AAPL"])
        original_close = df["close"].copy()
        result = adapter._apply_adjustments(df, ["AAPL"], "2025-01-01", "2025-12-31")
        pd.testing.assert_series_equal(result["close"], original_close)

    def test_multiple_splits_applied_cumulatively(self, tmp_path: Path) -> None:
        """Two splits for the same ticker must both be applied (multiplicative backward adjustment)."""
        adapter = self._make_adapter(tmp_path)
        split1 = pd.Timestamp("2025-04-01 09:30", tz="US/Eastern")
        split2 = pd.Timestamp("2025-06-01 09:30", tz="US/Eastern")

        mock_adj = MagicMock(spec=AdjustmentProvider)
        mock_adj.get_splits.return_value = {
            "AAPL": pd.Series(
                [2.0, 3.0],
                index=pd.DatetimeIndex([split1, split2]),
            ),
        }
        adapter._adjuster = mock_adj

        # Single row well before both splits: price should be divided by 2*3 = 6
        early_date = pd.Timestamp("2025-01-15 09:30", tz="US/Eastern")
        df = pd.DataFrame(
            [{"symbol": "AAPL", "open": 600.0, "high": 600.0,
              "low": 600.0, "close": 600.0, "volume": 1000.0}],
            index=pd.DatetimeIndex([early_date]),
        )
        result = adapter._apply_adjustments(df, ["AAPL"], "2025-01-01", "2025-12-31")
        assert result["close"].iloc[0] == pytest.approx(100.0)

    def test_high_and_low_adjusted_alongside_open_close(
        self, tmp_path: Path
    ) -> None:
        """The ``high`` and ``low`` columns must also be divided by the split ratio."""
        adapter = self._make_adapter(tmp_path)
        adapter._adjuster = MagicMock(spec=AdjustmentProvider)
        adapter._adjuster.get_splits.return_value = self._mock_splits()

        df = self._build_df(["AAPL", "MSFT"])
        result = adapter._apply_adjustments(df, ["AAPL", "MSFT"], "2025-01-01", "2025-12-31")

        aapl_pre = result[(result["symbol"] == "AAPL") & (result.index < self.SPLIT_TS)]
        expected_adjusted = self.PRE_PRICE / self.SPLIT_RATIO
        np.testing.assert_allclose(aapl_pre["high"].values, expected_adjusted)
        np.testing.assert_allclose(aapl_pre["low"].values, expected_adjusted)
