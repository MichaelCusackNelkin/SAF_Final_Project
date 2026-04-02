
#########################
#     Data Adapters     #
#########################

import os
import json
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Union
import pandas as pd
import numpy as np
import databento as db
from tqdm import tqdm
import yfinance as yf

def read_data_FRED(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.index = data['observation_date']
    data.drop(['observation_date'], axis=1, inplace = True)
    return data

class DataAdapter(ABC):
    """
    Abstract Base Class defining the standard interface for all data sources.
    Includes shared caching logic.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            os.makedirs(self.cache_dir)

    @abstractmethod
    def get_data(self, tickers: Union[str, Iterable[str], Enum, Iterable[Enum]],
                 start_date: str,
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetches data for the given tickers within the date range.
        Args:
            tickers: Single identifier or list of identifiers (str or Enum).
            start_date (str): YYYY-MM-DD start date.
            end_date (str, optional): YYYY-MM-DD end date. Defaults to current date.

        Returns:
            pd.DataFrame: DataFrame with DateTime index.
        """
        pass

    def _get_current_date(self) -> str:
        """Helper to get current date as string."""
        return datetime.today().strftime('%Y-%m-%d')

    def _generate_cache_key(self, tickers: Union[str, Iterable[str], Enum, Iterable[Enum]],
                            start_date: str, end_date: str) -> str:
        """
        Creates a deterministic hash based on tickers and date range.
        Handles both strings and Enums.
        """
        # Normalize tickers to a list
        if isinstance(tickers, (str, Enum)):
            ticker_list = [tickers]
        else:
            ticker_list = list(tickers)

        # Convert Enums to values or strings, ensure list of strings for hashing
        normalized_tickers = []
        for t in ticker_list:
            if isinstance(t, Enum):
                normalized_tickers.append(str(t.value))
            else:
                normalized_tickers.append(str(t))

        # Sort to ensure order doesn't affect hash
        normalized_tickers.sort()

        payload = {
            "tickers": normalized_tickers,
            "start": start_date,
            "end": end_date
        }
        # Serialize to JSON string ensuring deterministic order
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.md5(payload_str.encode('utf-8')).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Returns the full path for a cache file."""
        return self.cache_dir / f"{cache_key}.csv"

    def clear_cache(self):
        """Removes all CSV files in the cache directory."""
        for file in self.cache_dir.glob("*.csv"):
            file.unlink()
class YFinanceAdapter(DataAdapter):
    """
    Adapter for Yahoo Finance data (US-listed stocks and ETFs).
    Uses shared caching logic from DataAdapter.

    Caching behaviour:
        - Identical (tickers, start_date, end_date) → served from cache.
        - Any parameter change → new remote query, new cache file.
          Previous cache files are never modified or destroyed.
        - If the requested date range extends beyond today, a ValueError
          is raised and no data is returned.
    """

    def __init__(self, cache_dir: str = "data/yfinance_cache"):
        super().__init__(cache_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_tickers(
        self,
        tickers: Union[str, Iterable[str], Enum, Iterable[Enum]],
    ) -> list[str]:
        """Return a sorted list of uppercase ticker strings."""
        if isinstance(tickers, (str, Enum)):
            ticker_list = [tickers]
        else:
            ticker_list = list(tickers)

        normalized: list[str] = []
        for t in ticker_list:
            if isinstance(t, Enum):
                normalized.append(str(t.value).upper())
            else:
                normalized.append(str(t).upper())

        return sorted(set(normalized))

    def _validate_dates(self, start_date: str, end_date: str) -> None:
        """
        Validate the requested date window.

        Raises:
            ValueError: If *either* date is in the future, or if
                        start_date > end_date.
        """
        today = datetime.today().date()
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(
                f"Dates must be in YYYY-MM-DD format. Got start_date={start_date!r}, "
                f"end_date={end_date!r}."
            ) from exc

        if end_dt > today:
            raise ValueError(
                f"end_date ({end_date}) exceeds today's date ({today}). "
                "Cannot request future data."
            )
        if start_dt > today:
            raise ValueError(
                f"start_date ({start_date}) exceeds today's date ({today}). "
                "Cannot request future data."
            )
        if start_dt > end_dt:
            raise ValueError(
                f"start_date ({start_date}) is after end_date ({end_date})."
            )

    def _fetch_from_yfinance(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Execute the remote yfinance download and return a tidy DataFrame.

        Returns a DataFrame with a DatetimeIndex and a MultiIndex on
        columns (Price field, Ticker) when multiple tickers are requested,
        or simple column names for a single ticker.
        """
        data: pd.DataFrame = yf.download(
            tickers=" ".join(tickers),
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            return data

        # yfinance returns flat columns for a single ticker and MultiIndex
        # columns for multiple tickers.  Normalise to MultiIndex so the
        # consumer always gets a consistent shape.
        if len(tickers) == 1 and not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_product(
                [data.columns, tickers],
                names=["Price", "Ticker"],
            )

        return data

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_data(
        self,
        tickers: Union[str, Iterable[str], Enum, Iterable[Enum]],
        start_date: str,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch daily price data for US-listed stocks / ETFs.

        Args:
            tickers: One or more ticker symbols (str, Enum, or iterables
                     thereof).  Enum values are resolved via ``.value``.
            start_date: Inclusive start date in ``YYYY-MM-DD`` format.
            end_date: Inclusive end date in ``YYYY-MM-DD`` format.
                      Defaults to today.
            force_refresh: If ``True``, bypass the cache and always hit
                           the remote API.  A *new* cache file is written
                           (previous files are untouched).

        Returns:
            pd.DataFrame with a DatetimeIndex.

        Raises:
            ValueError: If any part of the requested date range is in the
                        future, or if dates are malformed.
        """
        # --- Resolve defaults ------------------------------------------------
        if end_date is None:
            end_date = self._get_current_date()

        # --- Validate dates (must happen before any data is returned) ---------
        self._validate_dates(start_date, end_date)

        # --- Normalise tickers ------------------------------------------------
        ticker_strings = self._normalize_tickers(tickers)

        # --- Cache lookup -----------------------------------------------------
        cache_key = self._generate_cache_key(ticker_strings, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        if not force_refresh and cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True, header=[0, 1])
            return df

        # --- Remote fetch -----------------------------------------------------
        df = self._fetch_from_yfinance(ticker_strings, start_date, end_date)

        # --- Persist to cache (new file; never overwrite existing ones) -------
        if not df.empty:
            df.to_csv(cache_path)

        return df

class DataBentoBarAdapter(DataAdapter):
    """
    Adapter for Databento intraday market data.

    Data flows through three layers of caching:

    1. Parquet cache (cache_dir/): Final processed DataFrames keyed by
       (tickers, dates, schema, freq). Cheapest to load.

    2. Raw DBN cache (cache_dir/raw_dbn/): Original .dbn.zst files from
       the Databento API. Re-processing is free; re-downloading costs money.

    3. Databento API (remote): Only hit when raw cache misses.

    Workflow:
        get_data() -> parquet cache -> raw DBN cache -> API fetch
        -> _process_intraday() -> _apply_adjustments() -> save parquet
    """

    def __init__(self, cache_dir: str = 'data/dbn_cache/bars', api_key: str = None,
                 verbose: bool = True):
        super().__init__(cache_dir)
        self.client = db.Historical(api_key)
        self.verbose = verbose

        # Separate folder for raw DBN downloads to avoid re-paying for data
        self.raw_data_dir = self.cache_dir / "raw_dbn"
        if not self.raw_data_dir.exists():
            os.makedirs(self.raw_data_dir)
    
    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _get_cache_path(self, cache_key: str) -> Path:
        """Parquet instead of CSV — faster for large intraday data."""
        return self.cache_dir / f"{cache_key}.parquet"

    def _get_raw_path(self, cache_key: str) -> Path:
        """Path for the raw .dbn.zst file from Databento."""
        return self.raw_data_dir / f"{cache_key}.dbn.zst"

    def _normalize_tickers(self, tickers: Union[str, Iterable[str]]) -> list[str]:
        """Ensure tickers is always a sorted list of uppercase strings."""
        if isinstance(tickers, str):
            ticker_list = [tickers]
        else:
            ticker_list = list(tickers)
        return sorted(set(t.upper() for t in ticker_list))

    def _generate_cache_key(self, tickers, start_date, end_date,
                            dataset="EQUS.MINI", schema="ohlcv-1m", freq="5min"):
        """
        Override base class to include dataset/schema/freq in the key.
        Same tickers+dates can produce different data depending on these.
        """
        if isinstance(tickers, str):
            ticker_list = [tickers]
        else:
            ticker_list = sorted(str(t) for t in tickers)

        payload = json.dumps({
            "tickers": ticker_list,
            "start": start_date,
            "end": end_date,
            "dataset": dataset,
            "schema": schema,
            "freq": freq,
        }, sort_keys=True)
        return hashlib.md5(payload.encode('utf-8')).hexdigest()

    def clear_cache(self):
        """Removes all processed parquet files. Raw .dbn.zst files are preserved."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()

    def _is_ohlcv_schema(self, schema: str) -> bool:
        """Check if the schema is an OHLCV variant (e.g. ohlcv-1m, ohlcv-1s)."""
        return schema.startswith("ohlcv")

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------
    def _fetch_databento(self, tickers: list[str], start_date: str,
                         end_date: str, dataset: str, schema: str,
                         raw_path: Path) -> db.DBNStore:
        """
        Hit the Databento API and stream the result to a .dbn.zst file.

        The `path` param tells the SDK to save the raw binary data to disk
        so we never have to pay to re-download the same data again.
        Returns a DBNStore object (call .to_df() to get a DataFrame).
        """
        data = self.client.timeseries.get_range(
            dataset=dataset,
            symbols=tickers,
            schema=schema,
            start=start_date,
            end=end_date,
            path=str(raw_path),
        )
        return data

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_data(self,
                 tickers: Union[str, Iterable[str]],
                 start_date: str,
                 end_date: Optional[str] = None,
                 dataset: str = "EQUS.MINI",
                 schema: str = "ohlcv-1m",
                 resample_freq: Optional[str] = None,
                 cache_only: bool = True,
                 raw_dbn: bool = False) -> Union[pd.DataFrame, db.DBNStore]:
        """
        Fetch intraday data for any Databento schema, processed and cached.

        For OHLCV schemas (ohlcv-1m, ohlcv-1s, etc.): resamples to
        resample_freq (default "1min") and applies split adjustments.

        For all other schemas (mbp-1, bbo-1s, trades, etc.): converts
        timezone to US/Eastern and filters to market hours only — no
        resampling or column assumptions.

        Args:
            tickers: One or more ticker symbols.
            start_date: Inclusive start (YYYY-MM-DD).
            end_date: Exclusive end (YYYY-MM-DD). Defaults to today.
            dataset: Databento dataset code (default "EQUS.MINI").
            schema: Any Databento schema string (default "ohlcv-1m").
            resample_freq: Pandas freq string to resample to (e.g. "5min").
                Only used for OHLCV schemas. Defaults to "1min" for OHLCV,
                ignored for other schemas.
            cache_only: If True (default), never hit the paid API — only
                        return data already cached. Set False to download.
            raw_dbn: If True, return the raw DBNStore object from cache
                instead of a processed DataFrame. Useful for replay().

        Returns:
            pd.DataFrame with US/Eastern DatetimeIndex, or db.DBNStore
            if raw_dbn=True.
        """
        if end_date is None:
            end_date = self._get_current_date()

        is_ohlcv = self._is_ohlcv_schema(schema)

        # Default resample_freq to "1min" for OHLCV schemas only
        if resample_freq is None and is_ohlcv:
            resample_freq = "1min"

        ticker_list = self._normalize_tickers(tickers)
        desc = f"BarAdapter {schema} {','.join(ticker_list)}"

        # -- Raw DBN shortcut: skip parquet cache entirely --
        if raw_dbn:
            raw_key = self._generate_cache_key(
                ticker_list, start_date, end_date, dataset, schema, freq="raw")
            raw_path = self._get_raw_path(raw_key)

            if raw_path.exists():
                return db.DBNStore.from_file(raw_path)
            elif not cache_only:
                return self._fetch_databento(
                    ticker_list, start_date, end_date, dataset, schema, raw_path)
            else:
                raise FileNotFoundError(
                    f"No cached raw DBN data for {ticker_list} ({start_date} to {end_date}). "
                    "Set cache_only=False to download from Databento."
                )

        steps = tqdm(total=4, desc=desc, unit="step", disable=not self.verbose)

        # -- 1. Check processed parquet cache --
        freq_for_key = resample_freq if resample_freq else "none"
        cache_key = self._generate_cache_key(
            ticker_list, start_date, end_date, dataset, schema, freq_for_key)
        cache_path = self._get_cache_path(cache_key)

        steps.set_postfix_str("checking parquet cache")
        if cache_path.exists():
            steps.update(4)
            steps.set_postfix_str("parquet cache hit")
            steps.close()
            return pd.read_parquet(cache_path)

        # -- 2. Check raw DBN cache --
        # Raw key uses freq="raw" since the .dbn.zst is at original resolution
        raw_key = self._generate_cache_key(
            ticker_list, start_date, end_date, dataset, schema, freq="raw")
        raw_path = self._get_raw_path(raw_key)

        steps.set_postfix_str("checking raw DBN cache")
        if raw_path.exists():
            # Free — just re-read the file we already downloaded
            data = db.DBNStore.from_file(raw_path)
        elif not cache_only:
            # -- 3. Hit Databento API (costs money) --
            steps.set_postfix_str("downloading from API")
            data = self._fetch_databento(
                ticker_list, start_date, end_date, dataset, schema, raw_path)
        else:
            steps.close()
            raise FileNotFoundError(
                f"No cached data for {ticker_list} ({start_date} to {end_date}). "
                "Set cache_only=False to download from Databento."
            )
        steps.update(1)

        # -- 4. Convert DBNStore -> DataFrame --
        steps.set_postfix_str("converting to DataFrame")
        df = data.to_df()
        steps.update(1)

        # -- 5. Process based on schema type --
        if is_ohlcv:
            steps.set_postfix_str("resampling + adjusting")
            df = self._process_intraday(df, freq=resample_freq)
            df = self._apply_adjustments(df, ticker_list, start_date, end_date)
        else:
            steps.set_postfix_str("processing")
            df = self._process_generic(df)
        steps.update(1)

        # -- 7. Save processed result to parquet --
        steps.set_postfix_str("saving parquet")
        df.to_parquet(cache_path)
        steps.update(1)
        steps.set_postfix_str("done")
        steps.close()

        return df

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------
    def _process_intraday(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Clean up raw intraday data:
        1. Convert timestamps from UTC -> US/Eastern.
        2. Filter to regular trading hours [09:30, 16:00).
        3. Resample to target frequency with standard OHLCV aggregation.
        """
        # to_df() sets ts_event as the index, already tz-aware in UTC
        df.index = df.index.tz_convert("US/Eastern")

        # Keep only regular trading hours
        df = df.between_time("09:30", "16:00", inclusive="left")

        # Resample per symbol to target frequency
        ts_col = df.index.name or "ts_event"
        df = (
            df.groupby("symbol")
            .resample(freq, include_groups=False)
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna(subset=["open"])
            .reset_index()
            .sort_values(by=[ts_col, "symbol"])
            .set_index(ts_col)
        )

        return df

    def _process_generic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Minimal processing for non-OHLCV schemas:
        1. Convert timestamps from UTC -> US/Eastern.
        2. Filter to regular trading hours [09:30, 16:00).

        No resampling or column assumptions — returns the raw columns
        from whatever Databento schema was requested.
        """
        df.index = df.index.tz_convert("US/Eastern")
        df = df.between_time("09:30", "16:00", inclusive="left")
        return df

    def _apply_adjustments(self, df: pd.DataFrame, tickers: list[str],
                           start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch stock splits from Yahoo Finance and apply them backward
        so historical prices are comparable to current prices.

        For each split, all prices BEFORE the split date are divided by
        the ratio and volume is multiplied by it.
        """
        import yfinance as yf

        for ticker in tickers:
            yf_ticker = yf.Ticker(ticker)
            splits = yf_ticker.splits

            if splits.empty:
                continue

            # Filter to our date range
            splits = splits.loc[start_date:end_date]

            for split_date, ratio in splits.items():
                if ratio == 0:
                    continue
                mask = df.index < split_date.tz_convert("US/Eastern")
                df.loc[mask, ["open", "high", "low", "close"]] /= ratio
                df.loc[mask, "volume"] *= ratio

        return df

class DataBentoMBP1Adapter(DataAdapter):
    """
    Adapter for Databento MBP-1 (top-of-book / L1) data.

    MBP-1 provides tick-level trade and quote events with best bid/ask.
    This adapter processes the raw tick data into:
      - Trade-only OHLCV bars with spread metrics (default)
      - Raw tick data with computed spread (if resample_freq=None)

    Accepts any schema that is a subset of MBP-1 (e.g. "tbbo", "bbo-1s",
    "bbo-1m"). All subset schemas produce the same columns as MBP-1.

    Uses the same three-layer caching as DataBentoAdapter:
      1. Parquet cache -> 2. Raw DBN cache -> 3. Databento API
    """

    MBP1_SUBSET_SCHEMAS = {"mbp-1", "tbbo", "bbo-1s", "bbo-1m"}

    def __init__(self, cache_dir: str = None,
                 api_key: str = None, schema: str = "mbp-1",
                 verbose: bool = True):
        if schema not in self.MBP1_SUBSET_SCHEMAS:
            raise ValueError(
                f"Schema {schema!r} is not a subset of MBP-1. "
                f"Accepted schemas: {self.MBP1_SUBSET_SCHEMAS}"
            )
        self.schema = schema
        self.verbose = verbose
        if cache_dir is None:
            cache_dir = f"data/dbn_cache/{schema.replace('-', '')}"
        super().__init__(cache_dir)
        self.client = db.Historical(api_key)
        self.raw_data_dir = self.cache_dir / "raw_dbn"
        if not self.raw_data_dir.exists():
            os.makedirs(self.raw_data_dir)

    def clear_cache(self):
        """Removes all processed parquet files. Raw .dbn.zst files are preserved."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.parquet"

    def _get_raw_path(self, cache_key: str) -> Path:
        return self.raw_data_dir / f"{cache_key}.dbn.zst"

    def _normalize_tickers(self, tickers: Union[str, Iterable[str]]) -> list[str]:
        if isinstance(tickers, str):
            ticker_list = [tickers]
        else:
            ticker_list = list(tickers)
        return sorted(set(t.upper() for t in ticker_list))

    def _generate_cache_key(self, tickers, start_date, end_date,
                            dataset="EQUS.MINI", freq="5min"):
        if isinstance(tickers, str):
            ticker_list = [tickers]
        else:
            ticker_list = sorted(str(t) for t in tickers)

        payload = json.dumps({
            "tickers": ticker_list,
            "start": start_date,
            "end": end_date,
            "dataset": dataset,
            "schema": self.schema,
            "freq": freq,
        }, sort_keys=True)
        return hashlib.md5(payload.encode('utf-8')).hexdigest()

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------
    def _fetch_databento(self, tickers: list[str], start_date: str,
                         end_date: str, dataset: str,
                         raw_path: Path) -> db.DBNStore:
        data = self.client.timeseries.get_range(
            dataset=dataset,
            symbols=tickers,
            schema=self.schema,
            start=start_date,
            end=end_date,
            path=str(raw_path),
        )
        return data

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_data(self,
                 tickers: Union[str, Iterable[str]],
                 start_date: str,
                 end_date: Optional[str] = None,
                 dataset: str = "EQUS.MINI",
                 resample_freq: Optional[str] = "5min",
                 trades_only: bool = True,
                 cache_only: bool = True,
                 raw_dbn: bool = False) -> Union[pd.DataFrame, db.DBNStore]:
        """
        Fetch MBP-1 data, optionally resampled into OHLCV bars with spread.

        Args:
            tickers: One or more ticker symbols.
            start_date: Inclusive start (YYYY-MM-DD).
            end_date: Exclusive end (YYYY-MM-DD). Defaults to today.
            dataset: Databento dataset code (default "EQUS.MINI").
            resample_freq: Pandas freq string (e.g. "5min", "1min").
                Set to None to return raw tick data.
            trades_only: If True, filter to trade events (action=="T")
                before resampling. Ignored when resample_freq is None.
            cache_only: If True (default), never hit the paid API.
            raw_dbn: If True, return the raw DBNStore object from cache
                instead of a processed DataFrame. Useful for replay().

        Returns:
            pd.DataFrame with US/Eastern DatetimeIndex, or db.DBNStore
            if raw_dbn=True.
        """
        if end_date is None:
            end_date = self._get_current_date()

        ticker_list = self._normalize_tickers(tickers)
        desc = f"{self.schema} {','.join(ticker_list)}"

        # -- Raw DBN shortcut: skip parquet cache entirely --
        if raw_dbn:
            raw_key = self._generate_cache_key(
                ticker_list, start_date, end_date, dataset, freq="raw")
            raw_path = self._get_raw_path(raw_key)

            if raw_path.exists():
                return db.DBNStore.from_file(raw_path)
            elif not cache_only:
                return self._fetch_databento(
                    ticker_list, start_date, end_date, dataset, raw_path)
            else:
                raise FileNotFoundError(
                    f"No cached raw {self.schema} data for {ticker_list} ({start_date} to {end_date}). "
                    "Set cache_only=False to download from Databento."
                )

        freq_for_key = resample_freq if resample_freq else "raw_ticks"
        steps = tqdm(total=4, desc=desc, unit="step", disable=not self.verbose)

        # -- 1. Check processed parquet cache --
        cache_key = self._generate_cache_key(
            ticker_list, start_date, end_date, dataset, freq_for_key)
        cache_path = self._get_cache_path(cache_key)

        steps.set_postfix_str("checking parquet cache")
        if cache_path.exists():
            steps.update(4)
            steps.set_postfix_str("parquet cache hit")
            steps.close()
            return pd.read_parquet(cache_path)

        # -- 2. Check raw DBN cache --
        raw_key = self._generate_cache_key(
            ticker_list, start_date, end_date, dataset, freq="raw")
        raw_path = self._get_raw_path(raw_key)

        steps.set_postfix_str("checking raw DBN cache")
        if raw_path.exists():
            data = db.DBNStore.from_file(raw_path)
        elif not cache_only:
            steps.set_postfix_str("downloading from API")
            data = self._fetch_databento(
                ticker_list, start_date, end_date, dataset, raw_path)
        else:
            steps.close()
            raise FileNotFoundError(
                f"No cached {self.schema} data for {ticker_list} ({start_date} to {end_date}). "
                "Set cache_only=False to download from Databento."
            )
        steps.update(1)

        # -- 3. Convert and process --
        steps.set_postfix_str("converting to DataFrame")
        df = data.to_df()
        steps.update(1)

        if resample_freq is not None:
            steps.set_postfix_str("resampling to bars")
            df = self._process_to_bars(df, ticker_list, resample_freq, trades_only)
        else:
            steps.set_postfix_str("processing raw ticks")
            df = self._process_raw(df, ticker_list)
        steps.update(1)

        # -- 4. Save processed result --
        steps.set_postfix_str("saving parquet")
        df.to_parquet(cache_path)
        steps.update(1)
        steps.set_postfix_str("done")
        steps.close()
        return df

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------
    def _process_to_bars(self, df: pd.DataFrame, tickers: list[str],
                         freq: str, trades_only: bool) -> pd.DataFrame:
        """
        Resample MBP-1 ticks into OHLCV bars with average spread.

        Filters to trade events (action=="T"), computes spread from
        best bid/ask, then resamples per symbol.
        """
        ts_col = df.index.name or "ts_event"
        df.index = df.index.tz_convert("US/Eastern")
        df = df.between_time("09:30", "16:00", inclusive="left")

        if df.empty:
            return pd.DataFrame()

        # Filter to requested tickers
        df = df[df["symbol"].isin(tickers)].copy()

        # Compute spread from top-of-book quotes
        df["spread"] = df["ask_px_00"] - df["bid_px_00"]

        if trades_only:
            df = df[df["action"] == "T"]

        if df.empty:
            return pd.DataFrame()

        result = (
            df.groupby("symbol")
            .resample(freq, include_groups=False)
            .agg(
                open=("price", "first"),
                high=("price", "max"),
                low=("price", "min"),
                close=("price", "last"),
                volume=("size", "sum"),
                avg_spread=("spread", "mean"),
            )
            .dropna(subset=["open"])
            .reset_index()
            .sort_values(by=[ts_col, "symbol"])
            .set_index(ts_col)
        )
        return result

    def _process_raw(self, df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """
        Minimal processing for raw tick-level MBP-1 data.
        Converts timezone, filters to market hours and requested tickers,
        and adds computed spread and book_imbalance columns.
        """
        df.index = df.index.tz_convert("US/Eastern")
        df = df.between_time("09:30", "16:00", inclusive="left")
        df = df[df["symbol"].isin(tickers)].copy()
        df["spread"] = df["ask_px_00"] - df["bid_px_00"]
        bid_sz = df["bid_sz_00"].replace(db.UNDEF_ORDER_SIZE, 0)
        ask_sz = df["ask_sz_00"].replace(db.UNDEF_ORDER_SIZE, 0)
        total_sz = bid_sz + ask_sz
        df["book_imbalance"] = (bid_sz.astype(np.int64) - ask_sz.astype(np.int64)) / total_sz.replace(0, np.nan)
        return df

class DataBentoMBP10Adapter(DataAdapter):
    """
    Adapter for Databento MBP-10 (market depth / L2) data.

    Returns raw tick-level data with all 10 depth levels plus a computed
    book_imbalance column:
      book_imbalance = (total_bid_sz - total_ask_sz) / (total_bid_sz + total_ask_sz)

    Uses the same three-layer caching as DataBentoAdapter:
      1. Parquet cache -> 2. Raw DBN cache -> 3. Databento API
    """

    N_LEVELS = 10

    def __init__(self, cache_dir: str = 'data/dbn_cache/mbp10', api_key: str = None,
                 verbose: bool = True):
        super().__init__(cache_dir)
        self.client = db.Historical(api_key)
        self.verbose = verbose
        self.raw_data_dir = self.cache_dir / "raw_dbn"
        if not self.raw_data_dir.exists():
            os.makedirs(self.raw_data_dir)

    def clear_cache(self):
        """Removes all processed parquet files. Raw .dbn.zst files are preserved."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.parquet"

    def _get_raw_path(self, cache_key: str) -> Path:
        return self.raw_data_dir / f"{cache_key}.dbn.zst"

    def _normalize_tickers(self, tickers: Union[str, Iterable[str]]) -> list[str]:
        if isinstance(tickers, str):
            ticker_list = [tickers]
        else:
            ticker_list = list(tickers)
        return sorted(set(t.upper() for t in ticker_list))

    def _generate_cache_key(self, tickers, start_date, end_date,
                            dataset="EQUS.MINI"):
        if isinstance(tickers, str):
            ticker_list = [tickers]
        else:
            ticker_list = sorted(str(t) for t in tickers)

        payload = json.dumps({
            "tickers": ticker_list,
            "start": start_date,
            "end": end_date,
            "dataset": dataset,
            "schema": "mbp-10",
        }, sort_keys=True)
        return hashlib.md5(payload.encode('utf-8')).hexdigest()

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------
    def _fetch_databento(self, tickers: list[str], start_date: str,
                         end_date: str, dataset: str,
                         raw_path: Path) -> db.DBNStore:
        data = self.client.timeseries.get_range(
            dataset=dataset,
            symbols=tickers,
            schema="mbp-10",
            start=start_date,
            end=end_date,
            path=str(raw_path),
        )
        return data

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_data(self,
                 tickers: Union[str, Iterable[str]],
                 start_date: str,
                 end_date: Optional[str] = None,
                 dataset: str = "EQUS.MINI",
                 cache_only: bool = True,
                 raw_dbn: bool = False) -> Union[pd.DataFrame, db.DBNStore]:
        """
        Fetch raw MBP-10 tick data with book_imbalance.

        Args:
            tickers: One or more ticker symbols.
            start_date: Inclusive start (YYYY-MM-DD).
            end_date: Exclusive end (YYYY-MM-DD). Defaults to today.
            dataset: Databento dataset code (default "EQUS.MINI").
            cache_only: If True (default), never hit the paid API.
            raw_dbn: If True, return the raw DBNStore object from cache
                instead of a processed DataFrame. Useful for replay().

        Returns:
            pd.DataFrame with US/Eastern DatetimeIndex containing all
            MBP-10 columns plus book_imbalance, or db.DBNStore if
            raw_dbn=True.
        """
        if end_date is None:
            end_date = self._get_current_date()

        ticker_list = self._normalize_tickers(tickers)
        desc = f"MBP10 {','.join(ticker_list)}"
        cache_key = self._generate_cache_key(
            ticker_list, start_date, end_date, dataset)

        # -- Raw DBN shortcut: skip parquet cache entirely --
        if raw_dbn:
            raw_path = self._get_raw_path(cache_key)

            if raw_path.exists():
                return db.DBNStore.from_file(raw_path)
            elif not cache_only:
                return self._fetch_databento(
                    ticker_list, start_date, end_date, dataset, raw_path)
            else:
                raise FileNotFoundError(
                    f"No cached raw MBP-10 data for {ticker_list} ({start_date} to {end_date}). "
                    "Set cache_only=False to download from Databento."
                )

        steps = tqdm(total=4, desc=desc, unit="step", disable=not self.verbose)

        # -- 1. Check processed parquet cache --
        cache_path = self._get_cache_path(cache_key)

        steps.set_postfix_str("checking parquet cache")
        if cache_path.exists():
            steps.update(4)
            steps.set_postfix_str("parquet cache hit")
            steps.close()
            return pd.read_parquet(cache_path)

        # -- 2. Check raw DBN cache --
        raw_path = self._get_raw_path(cache_key)

        steps.set_postfix_str("checking raw DBN cache")
        if raw_path.exists():
            data = db.DBNStore.from_file(raw_path)
        elif not cache_only:
            steps.set_postfix_str("downloading from API")
            data = self._fetch_databento(
                ticker_list, start_date, end_date, dataset, raw_path)
        else:
            steps.close()
            raise FileNotFoundError(
                f"No cached MBP-10 data for {ticker_list} ({start_date} to {end_date}). "
                "Set cache_only=False to download from Databento."
            )
        steps.update(1)

        # -- 3. Convert and process --
        steps.set_postfix_str("converting to DataFrame")
        df = data.to_df()
        steps.update(1)

        steps.set_postfix_str("computing book imbalance")
        df.index = df.index.tz_convert("US/Eastern")
        df = df.between_time("09:30", "16:00", inclusive="left")
        df = df[df["symbol"].isin(ticker_list)].copy()

        # Compute book imbalance across all 10 depth levels
        # Replace UNDEF_ORDER_SIZE sentinels (empty levels) with 0
        bid_sz_cols = [f"bid_sz_{i:02d}" for i in range(self.N_LEVELS)]
        ask_sz_cols = [f"ask_sz_{i:02d}" for i in range(self.N_LEVELS)]
        bid_sz = df[bid_sz_cols].replace(db.UNDEF_ORDER_SIZE, 0)
        ask_sz = df[ask_sz_cols].replace(db.UNDEF_ORDER_SIZE, 0)
        total_bid_sz = bid_sz.sum(axis=1)
        total_ask_sz = ask_sz.sum(axis=1)
        total_sz = total_bid_sz + total_ask_sz
        df["book_imbalance"] = (total_bid_sz.astype(np.int64) - total_ask_sz.astype(np.int64)) / total_sz.replace(0, np.nan)
        steps.update(1)

        # -- 4. Save processed result --
        steps.set_postfix_str("saving parquet")
        df.to_parquet(cache_path)
        steps.update(1)
        steps.set_postfix_str("done")
        steps.close()
        return df