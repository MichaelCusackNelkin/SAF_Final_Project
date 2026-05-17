from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DataAdapter(ABC):

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            os.makedirs(self.cache_dir)

    @abstractmethod
    def get_data(self, tickers, start_date, end_date=None) -> pd.DataFrame:
        pass

    def _get_current_date(self) -> str:
        return datetime.today().strftime("%Y-%m-%d")

    def _generate_cache_key(self, tickers, start_date: str, end_date: str,
                            interval: Optional[str] = None) -> str:
        if isinstance(tickers, (str, Enum)):
            ticker_list = [tickers]
        else:
            ticker_list = list(tickers)
        normalized = sorted(
            str(t.value) if isinstance(t, Enum) else str(t) for t in ticker_list
        )
        payload = {"tickers": normalized, "start": start_date, "end": end_date}
        if interval is not None:
            payload["interval"] = interval
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _get_cache_path(self, cache_key: str, format: str) -> Path:
        return self.cache_dir / f"{cache_key}.{format}"

# ── DataBentoAdapter ─────────────────────────────────────────────────────────

class DataBentoAdapter(DataAdapter):

    _POLL_INITIAL_SLEEP = 5.0
    _POLL_MAX_SLEEP = 60.0
    _POLL_BACKOFF = 1.5
    _POLL_TIMEOUT = 3600.0

    def __init__(self, cache_dir: str = "data/dbn_cache", api_key: str = None):
        super().__init__(cache_dir)
        self.client = db.Historical(api_key)
        self.raw_data_dir = self.cache_dir / "raw_dbn"
        if not self.raw_data_dir.exists():
            os.makedirs(self.raw_data_dir)

    def _get_cache_path(self, cache_key: str) -> Path:
        return super()._get_cache_path(cache_key=cache_key, format="parquet")

    def _generate_raw_cache_key(self, tickers: List[str], dataset: str,
                                schema: str, start_date: str, end_date: str) -> str:
        payload = {"tickers": sorted(tickers), "dataset": dataset,
                   "schema": schema, "start": start_date, "end": end_date}
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _get_raw_cache_path(self, raw_key: str) -> Path:
        return self.raw_data_dir / f"{raw_key}.dbn.zst"

    def get_data(self, tickers, start_date: str, end_date: Optional[str] = None,
                 dataset: str = "EQUS.MINI", schema: str = "ohlcv-1m",
                 resample_freq: Optional[str] = "5min", cache_only: bool = True,
                 local_files=None) -> pd.DataFrame:

        ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
        if end_date is None:
            end_date = self._get_current_date()
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)

        # L2 cache (final parquet)
        l2_key = self._generate_cache_key(ticker_list, start_date, end_date, resample_freq)
        l2_path = self._get_cache_path(l2_key)
        if l2_path.exists() and local_files is None:
            logger.info("L2 cache HIT: %s", l2_path.name)
            df = pd.read_parquet(l2_path)
            if df.index.tz is None:
                df.index = df.index.tz_localize("US/Eastern")
            return df

        # Local files bypass
        if local_files is not None:
            df = self._load_local_files(local_files)
        else:
            # L1 cache (raw DBN)
            raw_key = self._generate_raw_cache_key(ticker_list, dataset, schema,
                                                    start_date, end_date)
            raw_path = self._get_raw_cache_path(raw_key)

            if raw_path.exists():
                logger.info("L1 cache HIT: %s", raw_path.name)
                df = db.DBNStore.from_file(str(raw_path)).to_df()
            elif raw_path.with_suffix(".parquet").exists():
                logger.info("L1 cache HIT (merged): %s", raw_path.with_suffix(".parquet").name)
                df = pd.read_parquet(raw_path.with_suffix(".parquet"))
            else:
                logger.info("Cache MISS — querying DataBento...")
                result = self._fetch_from_databento(
                    ticker_list, dataset, schema, start_date, end_date, raw_path)
                df = result if isinstance(result, pd.DataFrame) else result.to_df()

        if df.empty:
            logger.warning("No data returned for %s", ticker_list)
            return df

        if "symbol" not in df.columns and "raw_symbol" not in df.columns:
            df["symbol"] = ticker_list[0]
        elif "raw_symbol" in df.columns:
            df.rename(columns={"raw_symbol": "symbol"}, inplace=True)

        df = self._apply_adjustments(df, ticker_list, start_date, end_date)
        df = self._process_intraday(df, freq=resample_freq)

        df.to_parquet(l2_path)
        logger.info("L2 cache saved: %s", l2_path.name)
        return df

    # ── Local file ingestion ──────────────────────────────────────────────

    def _load_local_files(self, local_files) -> pd.DataFrame:
        if isinstance(local_files, (str, Path)):
            file_list = [Path(local_files).expanduser().resolve()]
        else:
            file_list = [Path(f).expanduser().resolve() for f in local_files]

        for f in file_list:
            if not f.exists():
                raise FileNotFoundError(f"Local file not found: {f}")

        frames = []
        for f in file_list:
            fname = str(f)
            if fname.endswith((".dbn.zst", ".dbn")):
                frames.append(db.DBNStore.from_file(fname).to_df())
            elif fname.endswith((".csv", ".csv.zst")):
                frames.append(pd.read_csv(fname, parse_dates=True, index_col=0))
            else:
                raise ValueError(f"Unsupported format: {f.name}")

        df = pd.concat(frames, axis=0).sort_index()
        logger.info("Loaded %d rows from %d local file(s)", len(df), len(file_list))
        return df

    def ingest_to_cache(self, local_files, tickers, start_date: str, end_date: str,
                        dataset: str = "EQUS.MINI", schema: str = "ohlcv-1m") -> Path:
        """Register portal-downloaded files with L1 cache."""
        ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
        start_date, end_date = self._normalize_date(start_date), self._normalize_date(end_date)
        raw_key = self._generate_raw_cache_key(ticker_list, dataset, schema, start_date, end_date)
        raw_path = self._get_raw_cache_path(raw_key)

        if isinstance(local_files, (str, Path)):
            file_list = [Path(local_files).expanduser().resolve()]
        else:
            file_list = [Path(f).expanduser().resolve() for f in local_files]

        if len(file_list) == 1 and str(file_list[0]).endswith((".dbn.zst", ".dbn")):
            shutil.copy2(str(file_list[0]), str(raw_path))
            logger.info("Ingested → L1 cache: %s", raw_path.name)
            return raw_path

        df = self._load_local_files(file_list)
        merged_path = raw_path.with_suffix(".parquet")
        df.to_parquet(merged_path)
        logger.info("Ingested %d file(s) → L1 cache: %s", len(file_list), merged_path.name)
        return merged_path

    # ── DataBento batch API ───────────────────────────────────────────────

    def _fetch_from_databento(self, tickers, dataset, schema,
                              start_date, end_date, output_path):
        existing_id = self._find_existing_job(dataset, schema, tickers, start_date, end_date)
        if existing_id:
            logger.info("Reusing existing batch job: %s", existing_id)
            return self._download_job(existing_id, output_path)

        job = self.client.batch.submit_job(
            dataset=dataset, symbols=tickers, schema=schema,
            start=start_date, end=end_date, encoding="dbn", compression="zstd",
        )
        job_id = job["id"] if isinstance(job, dict) else job.id
        logger.info("Batch job submitted: %s", job_id)

        # Poll: valid states are received → queued → processing → finalizing → done | expired | purged
        sleep_time = self._POLL_INITIAL_SLEEP
        elapsed = 0.0
        while elapsed < self._POLL_TIMEOUT:
            time.sleep(sleep_time)
            elapsed += sleep_time

            current_state = None
            for j in self.client.batch.list_jobs():
                jid = j["id"] if isinstance(j, dict) else j.id
                if jid == job_id:
                    current_state = j["state"] if isinstance(j, dict) else j.state
                    break
            if current_state is None:
                raise RuntimeError(f"Job {job_id} not found")
            logger.info("  %s: %s (%.0fs)", job_id, current_state, elapsed)

            if current_state == "done":
                break
            if current_state in ("expired", "purged"):
                raise RuntimeError(f"Job {job_id} failed: {current_state}")
            sleep_time = min(sleep_time * self._POLL_BACKOFF, self._POLL_MAX_SLEEP)
        else:
            raise TimeoutError(f"Job {job_id} timed out after {self._POLL_TIMEOUT}s")

        return self._download_job(job_id, output_path)

    def _find_existing_job(self, dataset, schema, symbols, start, end):
        try:
            done_jobs = self.client.batch.list_jobs("done")
        except Exception:
            return None

        sorted_symbols = sorted(s.upper() for s in symbols)
        for j in done_jobs:
            d = j if isinstance(j, dict) else vars(j)
            j_syms = d.get("symbols", [])
            if isinstance(j_syms, str):
                j_syms = [j_syms]
            if (d.get("dataset") == dataset and d.get("schema") == schema
                    and sorted(s.upper() for s in j_syms) == sorted_symbols
                    and str(d.get("start", ""))[:10] == start[:10]
                    and str(d.get("end", ""))[:10] == end[:10]):
                return d.get("id")
        return None

    def _download_job(self, job_id, output_path):
        downloaded = self.client.batch.download(job_id=job_id, output_dir=str(output_path.parent))
        dbn_files = sorted(f for f in downloaded if str(f).endswith(".dbn.zst"))

        if not dbn_files:
            raise FileNotFoundError(f"No .dbn.zst files from job {job_id}: {downloaded}")

        if len(dbn_files) == 1:
            p = Path(dbn_files[0])
            if p != output_path:
                p.rename(output_path)
            return db.DBNStore.from_file(str(output_path))

        frames = []
        for f in dbn_files:
            frames.append(db.DBNStore.from_file(str(f)).to_df())
            Path(f).unlink(missing_ok=True)
        df = pd.concat(frames, axis=0).sort_index()
        df.to_parquet(output_path.with_suffix(".parquet"))
        return df

    # ── Split adjustment ──────────────────────────────────────────────────

    def _apply_adjustments(self, df: pd.DataFrame, tickers: List[str],
                           start_date: str, end_date: str) -> pd.DataFrame:
        """Backward split adjustment via yfinance. Prices ÷ ratio, volumes × ratio."""
        price_cols = ["open", "high", "low", "close"]

        for ticker in tickers:
            actions = yf.Ticker(ticker).actions
            if actions.empty or "Stock Splits" not in actions.columns:
                continue
            splits = actions[actions["Stock Splits"] > 0]["Stock Splits"]
            if splits.empty:
                continue

            if splits.index.tz is None:
                splits.index = splits.index.tz_localize("UTC")
            splits = splits[(splits.index >= pd.Timestamp(start_date, tz="UTC")) &
                            (splits.index <= pd.Timestamp(end_date, tz="UTC"))]
            if splits.empty:
                continue

            logger.info("Splits for %s: %s", ticker, splits.to_dict())
            ticker_mask = (df["symbol"].str.upper() == ticker.upper()
                           if "symbol" in df.columns
                           else pd.Series(True, index=df.index))

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            elif str(df.index.tz) != "UTC":
                df.index = df.index.tz_convert("UTC")

            for split_date, ratio in splits.sort_index(ascending=False).items():
                mask = ticker_mask & (df.index < split_date)
                for col in price_cols:
                    if col in df.columns:
                        df.loc[mask, col] = df.loc[mask, col] / ratio
                if "volume" in df.columns:
                    df.loc[mask, "volume"] = df.loc[mask, "volume"] * ratio

        return df

    # ── Market hours filter & resampling ──────────────────────────────────

    def _process_intraday(self, df: pd.DataFrame, freq: Optional[str] = "5min") -> pd.DataFrame:
        """Convert to ET, filter [09:30, 16:00), resample."""
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("US/Eastern")

        mkt_open = pd.Timestamp("09:30").time()
        mkt_close = pd.Timestamp("16:00").time()
        df = df[(df.index.time >= mkt_open) & (df.index.time < mkt_close)].copy()

        if df.empty or freq is None or freq == "1min":
            return df

        agg = {c: v for c, v in {"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}.items() if c in df.columns}
        if "symbol" in df.columns:
            df = (df.groupby("symbol").resample(freq).agg(agg)
                  .dropna(subset=["close"]).reset_index(level="symbol"))
        else:
            df = df.resample(freq).agg(agg).dropna(subset=["close"])

        df = df[(df.index.time >= mkt_open) & (df.index.time < mkt_close)]
        return df

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        d = date_str.strip()
        return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 and d.isdigit() else d