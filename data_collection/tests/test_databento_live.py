"""
test_databento_live.py

Unit tests for ``DataBentoLiveAdapter`` in
``data_collection.databento_live``.

All tests are isolated, deterministic, and never call the live Databento
API.  ``db.Live`` is fully mocked so no network connection is made.

Key design notes
----------------
- ``db.Live`` callbacks are driven synchronously from the test thread by
  calling ``adapter._on_record()`` directly.
- ``call_soon_threadsafe`` only works when the target event loop is
  running; queue-based tests run inside an ``asyncio`` coroutine with a
  real event loop using ``pytest.mark.asyncio``.
- With a single subscribed symbol the adapter emits a bar immediately
  after the first record arrives for a bucket (n_received == n_expected).
  Multi-symbol accumulation tests therefore use 2+ symbols and supply all
  of them per minute so the early-emit path triggers after both are seen.

Test areas
----------
- Dataclass construction: BarEvent, MultiSymbolBarEvent, LiveConfig
- Adapter construction: validation, env var API key fallback
- Bar aggregation: OHLCV correctness via ``_aggregate_minutes``
- Full pipeline: 2 symbols x 5 min -> MultiSymbolBarEvent on queue
- Partial bars: n_minutes reflects actual count
- Missing symbols: forward-filled; listed in missing_symbols
- Timestamp convention: left-labelled bucket, tz-aware
- Market hours filter: out-of-hours records dropped
- Instrument ID resolution: primary map, symbology_map fallback, unknown
- SymbolMappingMsg: populates _instrument_map
- Thread safety: concurrent map writes
- is_connected property
- stop(): resets flags, terminates client
- reconnect(): success/failure paths
- _FeedDropSentinel type identity
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import databento as db

from data_collection.databento_live import (
    BarEvent,
    DataBentoLiveAdapter,
    LiveConfig,
    MultiSymbolBarEvent,
    _FeedDropSentinel,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_KEY = "db-test-key-0000"
_SYMBOLS = ["AAPL", "MSFT"]

# A regular weekday in 2025 for all market-hours tests
_TRADE_DATE = "2025-03-10"


def _make_ohlcv(
    instrument_id: int,
    ts_et: pd.Timestamp,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: int = 1_000,
) -> db.OHLCVMsg:
    """
    Create a ``db.OHLCVMsg`` for a 1-minute bar whose close time is
    ``ts_et`` (given in US/Eastern).  Prices are stored as fixed-point
    integers (x1e9).

    Parameters
    ----------
    instrument_id : int
        Instrument identifier.
    ts_et : pd.Timestamp
        Bar close timestamp in US/Eastern.
    open_, high, low, close : float
        OHLCV prices.
    volume : int
        Bar volume.

    Returns
    -------
    db.OHLCVMsg
    """
    ts_utc = ts_et.tz_convert("UTC")
    ts_ns = int(ts_utc.timestamp() * 1e9)
    scale = int(1e9)
    return db.OHLCVMsg(
        rtype=32,
        publisher_id=1,
        instrument_id=instrument_id,
        ts_event=ts_ns,
        open=int(open_ * scale),
        high=int(high * scale),
        low=int(low * scale),
        close=int(close * scale),
        volume=volume,
    )


def _make_market_ts(date_str: str, hour: int, minute: int) -> pd.Timestamp:
    """
    Return a tz-aware US/Eastern ``pd.Timestamp``.

    Parameters
    ----------
    date_str : str
        Date in ``"YYYY-MM-DD"`` format.
    hour, minute : int
        Time components.

    Returns
    -------
    pd.Timestamp
    """
    return pd.Timestamp(f"{date_str} {hour:02d}:{minute:02d}:00", tz=ET)


async def _drain_queue_events(
    adapter: DataBentoLiveAdapter,
    queue: asyncio.Queue,
    records: list[db.OHLCVMsg],
    force_emit_bucket: pd.Timestamp | None = None,
) -> list[MultiSymbolBarEvent]:
    """
    Drive records through the adapter, optionally force-emit a bucket,
    yield to the event loop so ``call_soon_threadsafe`` callbacks fire,
    and return all ``MultiSymbolBarEvent`` objects from the queue.

    Parameters
    ----------
    adapter : DataBentoLiveAdapter
        The adapter instance.
    queue : asyncio.Queue
        The bar queue.
    records : list[db.OHLCVMsg]
        Records to deliver.
    force_emit_bucket : pd.Timestamp or None
        If set, cancel the grace timer and force-emit this bucket after
        delivering all records (for buckets that have not yet crossed a
        boundary).

    Returns
    -------
    list[MultiSymbolBarEvent]
    """
    for rec in records:
        adapter._on_record(rec)

    if force_emit_bucket is not None:
        adapter._cancel_grace_timer()
        with adapter._lock:
            if adapter._current_bucket == force_emit_bucket:
                adapter._emit_bucket(force_emit_bucket)

    # Give the event loop one iteration to process call_soon_threadsafe
    await asyncio.sleep(0)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def _make_adapter_with_real_loop(
    symbols: list[str] | None = None,
    config: LiveConfig | None = None,
) -> tuple[DataBentoLiveAdapter, asyncio.Queue]:
    """
    Build a ``DataBentoLiveAdapter`` using the *running* event loop.

    Must only be called from inside a coroutine (i.e. after the loop has
    started, e.g. inside an ``async def`` test).

    Parameters
    ----------
    symbols : list[str] or None
        Symbols to subscribe to.
    config : LiveConfig or None
        Optional config override.

    Returns
    -------
    tuple[DataBentoLiveAdapter, asyncio.Queue]
    """
    symbols = symbols or _SYMBOLS
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    adapter = DataBentoLiveAdapter(
        symbols=symbols,
        bar_queue=queue,
        loop=loop,
        config=config,
        api_key=_API_KEY,
    )
    adapter._running = True
    return adapter, queue


def _make_adapter_no_loop(
    symbols: list[str] | None = None,
    config: LiveConfig | None = None,
) -> tuple[DataBentoLiveAdapter, asyncio.Queue]:
    """
    Build a ``DataBentoLiveAdapter`` with a mock loop (no queue delivery).

    Useful for tests that only exercise internal state without needing
    the asyncio queue.

    Parameters
    ----------
    symbols : list[str] or None
        Symbols to subscribe to.
    config : LiveConfig or None
        Optional config override.

    Returns
    -------
    tuple[DataBentoLiveAdapter, asyncio.Queue]
    """
    symbols = symbols or _SYMBOLS
    queue: asyncio.Queue = asyncio.Queue()
    mock_loop = MagicMock()

    adapter = DataBentoLiveAdapter(
        symbols=symbols,
        bar_queue=queue,
        loop=mock_loop,
        config=config,
        api_key=_API_KEY,
    )
    adapter._running = True
    return adapter, queue


def _register_symbol(adapter: DataBentoLiveAdapter, iid: int, symbol: str) -> None:
    """
    Inject a symbol mapping into the adapter's instrument map.

    Parameters
    ----------
    adapter : DataBentoLiveAdapter
    iid : int
    symbol : str
    """
    with adapter._lock:
        adapter._instrument_map[iid] = symbol


# ---------------------------------------------------------------------------
# Section 1: Dataclass construction
# ---------------------------------------------------------------------------


class TestDataclassConstruction:
    """BarEvent, MultiSymbolBarEvent, and LiveConfig must be constructable."""

    def test_bar_event_constructable(self) -> None:
        ts = pd.Timestamp("2025-03-10 09:30:00", tz=ET)
        bar = BarEvent(
            timestamp=ts, symbol="AAPL", open=100.0, high=101.0,
            low=99.0, close=100.5, volume=1_000, n_minutes=5,
        )
        assert bar.symbol == "AAPL"
        assert bar.n_minutes == 5

    def test_multi_symbol_bar_event_constructable(self) -> None:
        ts = pd.Timestamp("2025-03-10 09:30:00", tz=ET)
        bar = BarEvent(
            timestamp=ts, symbol="AAPL", open=100.0, high=101.0,
            low=99.0, close=100.5, volume=500, n_minutes=5,
        )
        evt = MultiSymbolBarEvent(
            timestamp=ts,
            bars={"AAPL": bar},
            prices=pd.Series({"AAPL": 100.5}),
            volumes=pd.Series({"AAPL": 500.0}),
            n_minutes=5,
        )
        assert evt.n_minutes == 5
        assert evt.missing_symbols == []
        assert evt.is_stale is False

    def test_live_config_defaults(self) -> None:
        cfg = LiveConfig()
        assert cfg.dataset == "EQUS.MINI"
        assert cfg.live_schema == "ohlcv-1m"
        assert cfg.bar_interval_minutes == 5
        assert cfg.heartbeat_timeout_seconds == 90.0
        assert cfg.max_reconnect_retries == 10
        assert cfg.initial_backoff_seconds == 5.0
        assert cfg.max_backoff_seconds == 120.0
        assert 0 < cfg.min_symbols_for_bar <= 1.0
        assert cfg.grace_period_seconds > 0
        assert cfg.stale_threshold_seconds == 600.0


# ---------------------------------------------------------------------------
# Section 2: Adapter construction
# ---------------------------------------------------------------------------


class TestAdapterConstruction:
    """DataBentoLiveAdapter must validate inputs at construction time."""

    def test_basic_construction(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        assert adapter.symbols == ["AAPL", "MSFT"]

    def test_empty_symbols_raises(self) -> None:
        with pytest.raises(ValueError, match="symbols"):
            DataBentoLiveAdapter(
                symbols=[],
                bar_queue=asyncio.Queue(),
                api_key=_API_KEY,
            )

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            DataBentoLiveAdapter(
                symbols=["AAPL"],
                bar_queue=asyncio.Queue(),
            )

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABENTO_API_KEY", "env-key-123")
        adapter = DataBentoLiveAdapter(
            symbols=["AAPL"],
            bar_queue=asyncio.Queue(),
        )
        assert adapter._api_key == "env-key-123"

    def test_config_defaults_applied(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        assert adapter.config.bar_interval_minutes == 5


# ---------------------------------------------------------------------------
# Section 3: _aggregate_minutes — OHLCV correctness
# ---------------------------------------------------------------------------


class TestAggregateMinutes:
    """
    ``_aggregate_minutes`` must correctly compute OHLCV from a list of
    1-minute bar dicts.  These tests exercise the pure aggregation logic
    without going through the full record pipeline.
    """

    @staticmethod
    def _bars(*rows: tuple[float, float, float, float, int]) -> list[dict]:
        """Build a list of bar dicts from (open, high, low, close, volume) tuples."""
        return [
            {"open": o, "high": h, "low": l, "close": c, "volume": v}
            for o, h, l, c, v in rows
        ]

    def test_open_is_first_bar_open(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        ts = pd.Timestamp(f"{_TRADE_DATE} 09:30:00", tz=ET)
        bars = self._bars(
            (10.0, 12.0, 9.0, 11.0, 100),
            (11.5, 13.0, 10.0, 12.0, 200),
        )
        result = adapter._aggregate_minutes(ts, "AAPL", bars)
        assert result.open == pytest.approx(10.0)

    def test_high_is_max(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        ts = pd.Timestamp(f"{_TRADE_DATE} 09:30:00", tz=ET)
        bars = self._bars(
            (10.0, 12.0, 9.0, 11.0, 100),
            (11.5, 15.0, 10.0, 12.0, 200),
            (12.0, 11.0, 9.5, 10.5, 150),
        )
        result = adapter._aggregate_minutes(ts, "AAPL", bars)
        assert result.high == pytest.approx(15.0)

    def test_low_is_min(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        ts = pd.Timestamp(f"{_TRADE_DATE} 09:30:00", tz=ET)
        bars = self._bars(
            (10.0, 12.0, 9.0, 11.0, 100),
            (11.5, 13.0, 7.5, 12.0, 200),
        )
        result = adapter._aggregate_minutes(ts, "AAPL", bars)
        assert result.low == pytest.approx(7.5)

    def test_close_is_last_bar_close(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        ts = pd.Timestamp(f"{_TRADE_DATE} 09:30:00", tz=ET)
        bars = self._bars(
            (10.0, 12.0, 9.0, 11.0, 100),
            (11.5, 13.0, 10.0, 99.9, 200),
        )
        result = adapter._aggregate_minutes(ts, "AAPL", bars)
        assert result.close == pytest.approx(99.9)

    def test_volume_is_sum(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        ts = pd.Timestamp(f"{_TRADE_DATE} 09:30:00", tz=ET)
        bars = self._bars(
            (10.0, 12.0, 9.0, 11.0, 100),
            (11.5, 13.0, 10.0, 12.0, 250),
            (12.0, 11.0, 9.5, 10.5, 150),
        )
        result = adapter._aggregate_minutes(ts, "AAPL", bars)
        assert result.volume == 500

    def test_n_minutes_equals_len(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        ts = pd.Timestamp(f"{_TRADE_DATE} 09:30:00", tz=ET)
        bars = self._bars(
            (10.0, 12.0, 9.0, 11.0, 100),
            (11.5, 13.0, 10.0, 12.0, 200),
        )
        result = adapter._aggregate_minutes(ts, "AAPL", bars)
        assert result.n_minutes == 2

    def test_five_rows_n_minutes_five(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        ts = pd.Timestamp(f"{_TRADE_DATE} 09:30:00", tz=ET)
        bars = self._bars(*[(100.0, 101.0, 99.0, 100.5, 100)] * 5)
        result = adapter._aggregate_minutes(ts, "AAPL", bars)
        assert result.n_minutes == 5


# ---------------------------------------------------------------------------
# Section 4: Full pipeline — 2 symbols x 5 minutes -> queue event
# ---------------------------------------------------------------------------


class TestFullPipelineTwoSymbols:
    """
    5 minutes of records for 2 symbols must produce a ``MultiSymbolBarEvent``
    on the queue with correct OHLCV values.

    With 2 symbols, the adapter emits the bucket immediately after the
    second symbol's record arrives in each minute (early-emit: all symbols
    have reported).  After 5 minutes of such pairs, 5 separate bars will
    have been emitted — one per minute.  To test the 5-minute aggregated
    bar, we force-emit at the bucket boundary instead.

    Alternatively, we test the bucket accumulation by providing all 5
    minutes of data before the adapter processes a new-bucket transition.
    We achieve this by disabling the early-emit path (using a single-symbol
    scenario where the ``min_symbols_for_bar`` threshold is never reached
    until the grace period fires) or by driving the adapter to accumulate
    first.

    The cleanest approach: use the ``_emit_bucket`` method directly after
    accumulating all records via ``_handle_ohlcv``, bypassing ``_maybe_emit_bar``.
    """

    @pytest.mark.asyncio
    async def test_two_symbol_single_minute_bar_on_queue(self) -> None:
        adapter, queue = _make_adapter_with_real_loop(symbols=["AAPL", "MSFT"])
        _register_symbol(adapter, 1, "AAPL")
        _register_symbol(adapter, 2, "MSFT")

        # Both symbols report for minute 09:31 -> bucket 09:30
        ts_et = _make_market_ts(_TRADE_DATE, 9, 31)
        records = [
            _make_ohlcv(1, ts_et, open_=100.0, high=101.0, low=99.0, close=100.5, volume=500),
            _make_ohlcv(2, ts_et, open_=200.0, high=202.0, low=198.0, close=201.0, volume=1000),
        ]
        events = await _drain_queue_events(adapter, queue, records)

        assert len(events) == 1
        evt = events[0]
        assert "AAPL" in evt.bars
        assert "MSFT" in evt.bars
        assert evt.bars["AAPL"].volume == 500
        assert evt.bars["MSFT"].volume == 1000

    @pytest.mark.asyncio
    async def test_two_symbol_prices_series_correct(self) -> None:
        adapter, queue = _make_adapter_with_real_loop(symbols=["AAPL", "MSFT"])
        _register_symbol(adapter, 1, "AAPL")
        _register_symbol(adapter, 2, "MSFT")

        ts_et = _make_market_ts(_TRADE_DATE, 9, 31)
        records = [
            _make_ohlcv(1, ts_et, close=123.45),
            _make_ohlcv(2, ts_et, close=234.56),
        ]
        events = await _drain_queue_events(adapter, queue, records)

        assert len(events) == 1
        evt = events[0]
        assert evt.prices["AAPL"] == pytest.approx(123.45, abs=0.001)
        assert evt.prices["MSFT"] == pytest.approx(234.56, abs=0.001)

    @pytest.mark.asyncio
    async def test_five_minute_accumulated_bar(self) -> None:
        """
        Accumulate 5 minutes of records by inserting them directly into
        ``_minute_buffers`` (bypassing the early-emit), then force-emit.
        """
        adapter, queue = _make_adapter_with_real_loop(symbols=["AAPL"])
        _register_symbol(adapter, 1, "AAPL")

        bucket = pd.Timestamp(f"{_TRADE_DATE} 09:30:00", tz=ET)

        # Manually buffer 5 bars to sidestep the single-symbol early-emit
        minute_bars = [
            {"open": 100.0 + i * 0.1, "high": 101.0 + i * 0.1,
             "low": 99.0 - i * 0.05, "close": 100.5 + i * 0.1,
             "volume": 100 * (i + 1), "ts_event": bucket}
            for i in range(5)
        ]
        with adapter._lock:
            adapter._minute_buffers[(bucket, "AAPL")] = minute_bars
            adapter._current_bucket = bucket

        # Force emit
        with adapter._lock:
            adapter._emit_bucket(bucket)

        await asyncio.sleep(0)
        assert not queue.empty()
        event: MultiSymbolBarEvent = queue.get_nowait()
        bar = event.bars["AAPL"]

        assert bar.n_minutes == 5
        assert bar.open == pytest.approx(100.0, abs=0.001)   # first bar open
        assert bar.high == pytest.approx(101.4, abs=0.001)   # max high (i=4: 101.4)
        assert bar.low == pytest.approx(98.8, abs=0.001)     # min low (i=4: 99.0-4*0.05)
        assert bar.close == pytest.approx(100.9, abs=0.001)  # last bar close (i=4: 100.5+0.4)
        assert bar.volume == sum(100 * (i + 1) for i in range(5))

    @pytest.mark.asyncio
    async def test_two_symbol_volumes_series_correct(self) -> None:
        adapter, queue = _make_adapter_with_real_loop(symbols=["AAPL", "MSFT"])
        _register_symbol(adapter, 1, "AAPL")
        _register_symbol(adapter, 2, "MSFT")

        ts_et = _make_market_ts(_TRADE_DATE, 9, 31)
        records = [
            _make_ohlcv(1, ts_et, volume=300),
            _make_ohlcv(2, ts_et, volume=500),
        ]
        events = await _drain_queue_events(adapter, queue, records)

        assert len(events) == 1
        evt = events[0]
        assert evt.volumes["AAPL"] == pytest.approx(300.0)
        assert evt.volumes["MSFT"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Section 5: Partial bars
# ---------------------------------------------------------------------------


class TestPartialBars:
    """
    A bar emitted with fewer than ``bar_interval_minutes`` records must
    have ``n_minutes`` equal to the actual count.
    """

    @pytest.mark.asyncio
    async def test_partial_bar_n_minutes_three(self) -> None:
        adapter, queue = _make_adapter_with_real_loop(symbols=["AAPL"])
        _register_symbol(adapter, 1, "AAPL")

        bucket = pd.Timestamp(f"{_TRADE_DATE} 09:30:00", tz=ET)
        # Buffer only 3 bars manually
        with adapter._lock:
            adapter._minute_buffers[(bucket, "AAPL")] = [
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                 "volume": 100, "ts_event": bucket}
                for _ in range(3)
            ]
            adapter._current_bucket = bucket
            adapter._emit_bucket(bucket)

        await asyncio.sleep(0)
        event: MultiSymbolBarEvent = queue.get_nowait()
        assert event.bars["AAPL"].n_minutes == 3


# ---------------------------------------------------------------------------
# Section 6: Missing symbols (forward-fill)
# ---------------------------------------------------------------------------


class TestMissingSymbols:
    """
    Symbols that sent no data in a window must be forward-filled and
    listed in ``missing_symbols``.
    """

    @pytest.mark.asyncio
    async def test_missing_symbol_forward_filled(self) -> None:
        adapter, queue = _make_adapter_with_real_loop(symbols=["AAPL", "MSFT"])
        _register_symbol(adapter, 1, "AAPL")

        # Prime last close for MSFT
        with adapter._lock:
            adapter._last_close["MSFT"] = 250.0

        bucket = pd.Timestamp(f"{_TRADE_DATE} 09:30:00", tz=ET)
        # Only buffer AAPL
        with adapter._lock:
            adapter._minute_buffers[(bucket, "AAPL")] = [
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                 "volume": 200, "ts_event": bucket}
            ]
            adapter._current_bucket = bucket
            adapter._emit_bucket(bucket)

        await asyncio.sleep(0)
        event: MultiSymbolBarEvent = queue.get_nowait()
        assert "MSFT" in event.missing_symbols
        assert event.bars["MSFT"].close == pytest.approx(250.0)
        assert event.bars["MSFT"].volume == 0

    @pytest.mark.asyncio
    async def test_no_missing_symbols_when_all_report(self) -> None:
        adapter, queue = _make_adapter_with_real_loop(symbols=["AAPL", "MSFT"])
        _register_symbol(adapter, 1, "AAPL")
        _register_symbol(adapter, 2, "MSFT")

        ts_et = _make_market_ts(_TRADE_DATE, 9, 31)
        records = [
            _make_ohlcv(1, ts_et, close=100.0),
            _make_ohlcv(2, ts_et, close=200.0),
        ]
        events = await _drain_queue_events(adapter, queue, records)

        assert len(events) == 1
        assert events[0].missing_symbols == []


# ---------------------------------------------------------------------------
# Section 7: Timestamp and bucket conventions
# ---------------------------------------------------------------------------


class TestTimestampConventions:
    """
    Buckets must be left-labelled and the bucket timestamp must be tz-aware.
    """

    @pytest.mark.asyncio
    async def test_bucket_is_left_labelled_09_30(self) -> None:
        adapter, queue = _make_adapter_with_real_loop(symbols=["AAPL", "MSFT"])
        _register_symbol(adapter, 1, "AAPL")
        _register_symbol(adapter, 2, "MSFT")

        # Records at 09:31 ET (1-min bar close) -> bar open 09:30 -> bucket 09:30
        ts_et = _make_market_ts(_TRADE_DATE, 9, 31)
        records = [_make_ohlcv(1, ts_et), _make_ohlcv(2, ts_et)]
        events = await _drain_queue_events(adapter, queue, records)

        assert len(events) == 1
        bucket = events[0].timestamp
        assert bucket.hour == 9
        assert bucket.minute == 30

    @pytest.mark.asyncio
    async def test_bucket_timestamp_is_tz_aware(self) -> None:
        adapter, queue = _make_adapter_with_real_loop(symbols=["AAPL", "MSFT"])
        _register_symbol(adapter, 1, "AAPL")
        _register_symbol(adapter, 2, "MSFT")

        ts_et = _make_market_ts(_TRADE_DATE, 9, 31)
        records = [_make_ohlcv(1, ts_et), _make_ohlcv(2, ts_et)]
        events = await _drain_queue_events(adapter, queue, records)

        assert len(events) == 1
        assert events[0].timestamp.tzinfo is not None

    def test_compute_bucket_floors_to_interval(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        # 09:33 ET -> floor to 09:30 for 5-min interval
        ts = pd.Timestamp(f"{_TRADE_DATE} 09:33:00", tz=ET)
        bucket = adapter._compute_bucket(ts)
        assert bucket.hour == 9
        assert bucket.minute == 30

    def test_compute_bucket_09_35_floors_to_09_35(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        ts = pd.Timestamp(f"{_TRADE_DATE} 09:35:00", tz=ET)
        bucket = adapter._compute_bucket(ts)
        assert bucket.hour == 9
        assert bucket.minute == 35

    def test_compute_bucket_09_39_floors_to_09_35(self) -> None:
        adapter, _ = _make_adapter_no_loop()
        ts = pd.Timestamp(f"{_TRADE_DATE} 09:39:00", tz=ET)
        bucket = adapter._compute_bucket(ts)
        assert bucket.hour == 9
        assert bucket.minute == 35


# ---------------------------------------------------------------------------
# Section 8: Market hours filter
# ---------------------------------------------------------------------------


class TestMarketHoursFilter:
    """
    Records outside [09:30, 16:00) ET and on weekends must be dropped.
    """

    @pytest.mark.parametrize("hour,minute,expected", [
        (9, 30, True),
        (9, 31, True),
        (15, 59, True),
        (16, 0, False),
        (16, 1, False),
        (9, 29, False),
    ])
    def test_is_market_hours_boundaries(
        self, hour: int, minute: int, expected: bool,
    ) -> None:
        ts = _make_market_ts(_TRADE_DATE, hour, minute)
        assert DataBentoLiveAdapter._is_market_hours(ts) is expected

    def test_weekend_is_not_market_hours(self) -> None:
        ts = pd.Timestamp("2025-03-08 10:00:00", tz=ET)  # Saturday
        assert DataBentoLiveAdapter._is_market_hours(ts) is False

    def test_pre_market_records_not_buffered(self) -> None:
        adapter, _ = _make_adapter_no_loop(symbols=["AAPL"])
        _register_symbol(adapter, 1, "AAPL")

        ts_et = _make_market_ts(_TRADE_DATE, 9, 0)   # 09:00 pre-market
        adapter._on_record(_make_ohlcv(1, ts_et))

        with adapter._lock:
            assert len(adapter._minute_buffers) == 0

    def test_post_market_records_not_buffered(self) -> None:
        adapter, _ = _make_adapter_no_loop(symbols=["AAPL"])
        _register_symbol(adapter, 1, "AAPL")

        ts_et = _make_market_ts(_TRADE_DATE, 16, 1)  # 16:01 post-market
        adapter._on_record(_make_ohlcv(1, ts_et))

        with adapter._lock:
            assert len(adapter._minute_buffers) == 0


# ---------------------------------------------------------------------------
# Section 9: Instrument ID resolution
# ---------------------------------------------------------------------------


class TestInstrumentIdResolution:
    """
    Records must resolve via ``_instrument_map`` first, then
    ``symbology_map`` fallback, then be dropped if unknown.
    """

    def test_primary_instrument_map_resolves(self) -> None:
        adapter, _ = _make_adapter_no_loop(symbols=["AAPL"])
        _register_symbol(adapter, 99, "AAPL")

        ts_et = _make_market_ts(_TRADE_DATE, 9, 31)
        adapter._on_record(_make_ohlcv(99, ts_et))

        with adapter._lock:
            assert len(adapter._minute_buffers) > 0

    def test_symbology_map_fallback(self) -> None:
        adapter, _ = _make_adapter_no_loop(symbols=["AAPL"])

        # No entry in _instrument_map; provide a mock live client
        mock_client = MagicMock()
        mock_client.symbology_map = {77: "AAPL"}
        adapter._live_client = mock_client

        ts_et = _make_market_ts(_TRADE_DATE, 9, 31)
        adapter._on_record(_make_ohlcv(77, ts_et))

        with adapter._lock:
            assert adapter._instrument_map.get(77) == "AAPL"

    def test_unknown_instrument_id_skipped(self) -> None:
        adapter, _ = _make_adapter_no_loop(symbols=["AAPL"])
        adapter._live_client = None  # no fallback either

        ts_et = _make_market_ts(_TRADE_DATE, 9, 31)
        adapter._on_record(_make_ohlcv(999, ts_et))

        with adapter._lock:
            assert len(adapter._minute_buffers) == 0

    def test_symbol_mapping_msg_updates_instrument_map(self) -> None:
        adapter, _ = _make_adapter_no_loop(symbols=["AAPL"])
        ts_ns = int(pd.Timestamp(f"{_TRADE_DATE} 09:30:00", tz=UTC).timestamp() * 1e9)
        msg = db.SymbolMappingMsg(
            publisher_id=1,
            instrument_id=55,
            ts_event=ts_ns,
            stype_in=db.SType.RAW_SYMBOL,
            stype_in_symbol="AAPL",
            stype_out=db.SType.RAW_SYMBOL,
            stype_out_symbol="AAPL",
            start_ts=ts_ns,
            end_ts=ts_ns,
        )
        adapter._on_record(msg)
        with adapter._lock:
            assert adapter._instrument_map[55] == "AAPL"


# ---------------------------------------------------------------------------
# Section 10: Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """
    Concurrent writes to the symbol map from multiple threads must not
    raise exceptions or corrupt state.
    """

    def test_concurrent_symbol_registration_no_crash(self) -> None:
        n = 20
        adapter, _ = _make_adapter_no_loop(symbols=list(f"SYM{i}" for i in range(n)))
        errors: list[Exception] = []

        def _register(iid: int, sym: str) -> None:
            try:
                with adapter._lock:
                    adapter._instrument_map[iid] = sym
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=_register, args=(i, f"SYM{i}"))
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        with adapter._lock:
            assert len(adapter._instrument_map) == n


# ---------------------------------------------------------------------------
# Section 11: is_connected property
# ---------------------------------------------------------------------------


class TestIsConnected:
    """``is_connected`` must be True only when both flags are True."""

    @pytest.mark.parametrize("connected,running,expected", [
        (True,  True,  True),
        (True,  False, False),
        (False, True,  False),
        (False, False, False),
    ])
    def test_is_connected(
        self,
        connected: bool,
        running: bool,
        expected: bool,
    ) -> None:
        adapter, _ = _make_adapter_no_loop()
        adapter._connected = connected
        adapter._running = running
        assert adapter.is_connected is expected


# ---------------------------------------------------------------------------
# Section 12: stop() behaviour
# ---------------------------------------------------------------------------


class TestStopBehaviour:
    """After ``stop()`` the adapter must not be connected."""

    @pytest.mark.asyncio
    async def test_stop_sets_not_connected(self) -> None:
        adapter, _ = _make_adapter_with_real_loop()
        mock_client = MagicMock()
        adapter._live_client = mock_client
        adapter._connected = True
        adapter._running = True

        await adapter.stop()

        assert adapter._connected is False
        assert adapter._running is False

    @pytest.mark.asyncio
    async def test_stop_calls_terminate_on_client(self) -> None:
        adapter, _ = _make_adapter_with_real_loop()
        mock_client = MagicMock()
        adapter._live_client = mock_client
        adapter._running = True

        await adapter.stop()

        mock_client.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_tolerates_terminate_exception(self) -> None:
        adapter, _ = _make_adapter_with_real_loop()
        mock_client = MagicMock()
        mock_client.terminate.side_effect = RuntimeError("already closed")
        adapter._live_client = mock_client
        adapter._running = True

        await adapter.stop()  # must not raise

        assert adapter._connected is False


# ---------------------------------------------------------------------------
# Section 13: reconnect() behaviour
# ---------------------------------------------------------------------------


class TestReconnectBehaviour:
    """``reconnect()`` returns True on success, False on max-retry failure."""

    @pytest.mark.asyncio
    async def test_reconnect_success_returns_true(self) -> None:
        adapter, _ = _make_adapter_with_real_loop()
        adapter._running = False

        def _fake_create_and_start(start_param=None):
            adapter._connected = True
            adapter._running = True
            # Set _last_record_time far in the future so the confirm-wait exits
            adapter._last_record_time = time.monotonic() + 1000

        with patch.object(adapter, "_create_and_start_client", side_effect=_fake_create_and_start), \
             patch("asyncio.sleep", return_value=None):
            result = await adapter.reconnect()

        assert result is True

    @pytest.mark.asyncio
    async def test_reconnect_failure_returns_false(self) -> None:
        config = LiveConfig(
            max_reconnect_retries=2,
            initial_backoff_seconds=0.0,
        )
        adapter, _ = _make_adapter_with_real_loop(config=config)

        def _fail(*_args, **_kwargs):
            raise RuntimeError("connection refused")

        with patch.object(adapter, "_create_and_start_client", side_effect=_fail), \
             patch("asyncio.sleep", return_value=None):
            result = await adapter.reconnect()

        assert result is False

    @pytest.mark.asyncio
    async def test_reconnect_resets_backoff_on_success(self) -> None:
        adapter, _ = _make_adapter_with_real_loop()
        adapter._running = False
        adapter._current_backoff = 60.0  # inflated

        def _fake_create_and_start(start_param=None):
            adapter._connected = True
            adapter._running = True
            adapter._last_record_time = time.monotonic() + 1000

        with patch.object(adapter, "_create_and_start_client", side_effect=_fake_create_and_start), \
             patch("asyncio.sleep", return_value=None):
            await adapter.reconnect()

        assert adapter._current_backoff == adapter.config.initial_backoff_seconds


# ---------------------------------------------------------------------------
# Section 14: _FeedDropSentinel
# ---------------------------------------------------------------------------


class TestFeedDropSentinel:
    """``_FeedDropSentinel`` must be a distinct type for isinstance checks."""

    def test_sentinel_is_its_own_type(self) -> None:
        s = _FeedDropSentinel()
        assert isinstance(s, _FeedDropSentinel)
        assert not isinstance(s, MultiSymbolBarEvent)

    def test_sentinel_is_not_bar_event(self) -> None:
        s = _FeedDropSentinel()
        assert not isinstance(s, BarEvent)
