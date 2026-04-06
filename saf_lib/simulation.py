"""
Order book simulator
"""

import numpy as np
import pandas as pd
import databento as db
import polars as pl

class InvalidOrderBookError(Exception):
    def __init__(self, message):
        super().__init__(message)

class L2Book:
    """
    Class instantiation of a snapshot of an L2 orderbook at a signle point in time.
    """
    def __init__(self, snapshot_df):
        # --- Check Book Validity ---
        df_invalid = snapshot_df.filter(pl.col('ask_px') < pl.col("bid_px"))
        ask_greater_than_bid = df_invalid.shape[0] != 0
        if ask_greater_than_bid:
            raise InvalidOrderBookError(f"Ask greater than bid at level(s) {df_invalid['level'].to_list()}")
        # Only levels with non-zero size and non-zero prices
        snapshot_df = snapshot_df.filter(
            pl.col('ask_size') > 0,
            pl.col('bid_size') > 0,
            pl.col('ask_px') > 0,
            pl.col('bid_px') > 0
        )
        
        # --- Build Orderbook Snapshot ---
        self.bids = [(row['bid_px'], row['bid_size']) for row in snapshot_df.sort('level').iter_rows(named=True)]
        self.asks = [(row['ask_px'], row['ask_size']) for row in snapshot_df.sort('level').iter_rows(named=True)]
    
    def __str__(self):
        return f"Asks: {self.asks} \nBids: {self.bids}"
    
    def accept_mkt_order(self, side, order_size):
        if side == "BUY":
            book_temp = self.asks
        if side == "SELL":
            book_temp = self.bids
        
        volume_weighted_fills = []
        size_remaining = order_size
        levels_swept = 0
        for level, idx in zip(book_temp, range(len(book_temp))):
            level_px = level[0]
            level_size = level[1]

            # Filling Logic
            if size_remaining < level_size:
                volume_weighted_fills.append(size_remaining * level_px)
                book_temp[idx] = (level_px, level_size - size_remaining)
                size_remaining = 0
            elif size_remaining == level_size:
                volume_weighted_fills.append(size_remaining * level_px)
                book_temp[idx] = (level_px, level_size - size_remaining)
                size_remaining = 0
                levels_swept +=1
            elif size_remaining > level_size:
                volume_weighted_fills.append(level_size * level_px)
                size_remaining -= level_size
                levels_swept += 1            
            if size_remaining == 0:
                break

        # Remove swepts levels and re-assign    
        del book_temp[0:levels_swept]

        if side == "BUY":
            self.asks = book_temp
        if side == "SELL":
            self.bids = book_temp

        return sum(volume_weighted_fills) / order_size

class ExecutionSimulator:
    """
    Replay-driven execution simulator that walks a time-sorted L2 orderbook
    DataFrame and simulates market-order fills against the book state.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame with columns ``ts_event_nano``, ``instrument_id``,
        and 10 levels of ``bid_px_XX``, ``ask_px_XX``, ``bid_sz_XX``,
        ``ask_sz_XX`` (XX = 00..09).
    """

    _N_LEVELS: int = 10

    def __init__(self, df: pl.DataFrame) -> None:
        df = df.sort("ts_event_nano")

        # Extract numpy arrays for fast row-level access (writable copies).
        self.ts_ns: np.ndarray = df["ts_event_nano"].to_numpy().copy()
        self.instrument_id: np.ndarray = df["instrument_id"].to_numpy().copy()

        lvls = [f"{i:02d}" for i in range(self._N_LEVELS)]
        self.bid_px: np.ndarray = df.select([f"bid_px_{l}" for l in lvls]).to_numpy().copy().astype(np.float64)
        self.ask_px: np.ndarray = df.select([f"ask_px_{l}" for l in lvls]).to_numpy().copy().astype(np.float64)
        self.bid_sz: np.ndarray = df.select([f"bid_sz_{l}" for l in lvls]).to_numpy().copy().astype(np.float64)
        self.ask_sz: np.ndarray = df.select([f"ask_sz_{l}" for l in lvls]).to_numpy().copy().astype(np.float64)

        self._n_rows: int = len(self.ts_ns)
        self._cur_idx: int = 0

        # Internal book state — mutable lists of (price, size) tuples.
        self.bids: list[tuple[float, int]]
        self.asks: list[tuple[float, int]]
        self.bids, self.asks = self._build_book_from_row(0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_book_from_row(self, idx: int) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
        """
        Read row *idx* from the numpy arrays and return ``(bids, asks)``.

        Parameters
        ----------
        idx : int
            Row index into the numpy arrays.

        Returns
        -------
        tuple[list[tuple[float, int]], list[tuple[float, int]]]
            Bid and ask sides, each a list of ``(price, size)`` tuples with
            zero-price or zero-size levels filtered out.
        """
        bids = [
            (float(self.bid_px[idx, l]), int(self.bid_sz[idx, l]))
            for l in range(self._N_LEVELS)
            if self.bid_px[idx, l] > 0 and self.bid_sz[idx, l] > 0
        ]
        asks = [
            (float(self.ask_px[idx, l]), int(self.ask_sz[idx, l]))
            for l in range(self._N_LEVELS)
            if self.ask_px[idx, l] > 0 and self.ask_sz[idx, l] > 0
        ]
        return bids, asks
    
    def _sync_book_to_arrays(self) -> None:
        """Write ``self.bids`` / ``self.asks`` back into the numpy arrays at
        ``self._cur_idx``, shifting remaining levels to the front and zeroing
        out any trailing (now-empty) level slots."""
        idx = self._cur_idx

        # --- bids ---
        for l in range(self._N_LEVELS):
            if l < len(self.bids):
                self.bid_px[idx, l] = self.bids[l][0]
                self.bid_sz[idx, l] = self.bids[l][1]
            else:
                self.bid_px[idx, l] = np.nan
                self.bid_sz[idx, l] = np.nan

        # --- asks ---
        for l in range(self._N_LEVELS):
            if l < len(self.asks):
                self.ask_px[idx, l] = self.asks[l][0]
                self.ask_sz[idx, l] = self.asks[l][1]
            else:
                self.ask_px[idx, l] = np.nan
                self.ask_sz[idx, l] = np.nan

    def _apply_impact_to_future(self, side: str, fills: list[tuple[float, int]]) -> None:
        """
        Propagate liquidity removal from *fills* into all future rows.

        For each filled price level, every future row whose corresponding
        price array matches that price has its size reduced by the filled
        quantity (clamped to 0).

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"`` — determines which arrays to modify.
        fills : list[tuple[float, int]]
            List of ``(fill_price, fill_qty)`` produced by the execution.
        """
        start = self._cur_idx + 1
        if start >= self._n_rows or not fills:
            return

        if side == "BUY":
            px_arr = self.ask_px
            sz_arr = self.ask_sz
        else:
            px_arr = self.bid_px
            sz_arr = self.bid_sz

        future_px = px_arr[start:]
        future_sz = sz_arr[start:]

        for fill_px, fill_qty in fills:
            mask = future_px == fill_px
            future_sz[mask] -= fill_qty
        np.maximum(future_sz, 0, out=future_sz)

        # Compact: remove zero-size levels (where price is still set) and
        # shift deeper levels forward so there are no gaps.
        # A level needs removal when size==0 but price is not NaN (genuinely emptied).
        has_gap = np.any((future_sz == 0) & ~np.isnan(future_px), axis=1)
        for row in np.where(has_gap)[0]:
            keep = (future_sz[row] > 0) | np.isnan(future_px[row])
            n_keep = int(np.sum(keep))
            future_px[row, :n_keep] = future_px[row][keep]
            future_sz[row, :n_keep] = future_sz[row][keep]
            future_px[row, n_keep:] = np.nan
            future_sz[row, n_keep:] = np.nan

    # ------------------------------------------------------------------
    # time control
    # ------------------------------------------------------------------

    def advance_to(self, ts_nano: int) -> None:
        """
        Advance the simulator to the last snapshot with ``ts_event_nano <= ts_nano``.

        The internal book state is replaced by the market snapshot at that
        point in time, discarding any prior simulated modifications.

        Parameters
        ----------
        ts_nano : int
            Target nanosecond timestamp.
        """
        # searchsorted(..., "right") gives the first index *after* ts_nano;
        # subtract 1 to land on the last row <= ts_nano.
        target = int(np.searchsorted(self.ts_ns, ts_nano, side="right")) - 1
        if target < 0:
            target = 0
        if target >= self._n_rows:
            target = self._n_rows - 1

        self._cur_idx = target
        self.bids, self.asks = self._build_book_from_row(target)

    # ------------------------------------------------------------------
    # order execution
    # ------------------------------------------------------------------

    def accept_mkt_order(self, side: str, order_size: int, ts_nano: int, latency_ns: int = 0) -> dict:
        """
        Simulate a market order at a given timestamp.

        Parameters
        ----------
        side : str
            ``"BUY"`` (walks asks) or ``"SELL"`` (walks bids).
        order_size : int
            Number of shares to fill.
        ts_nano : int
            Nanosecond timestamp at which the order is sent.
        latency_ns : int
            Additional latency in nanoseconds before execution (default 0).

        Returns
        -------
        dict
            ``vwap`` — volume-weighted average fill price.
            ``fills`` — list of ``(price, qty_filled)`` tuples.
            ``total_filled`` — total shares filled (may be < *order_size*).
            ``levels_swept`` — number of levels fully consumed.
            ``executed_at_ns`` — ``ts_event_nano`` of the book row the order
            executed against.
            ``executed_at_idx`` — integer index of that row in the internal
            arrays (useful for slicing).
            ``arrival_cost`` — difference between VWAP and midprice at
            execution time (positive means paid more than mid for BUY,
            received less than mid for SELL).
        """
        self.advance_to(ts_nano + latency_ns)
        mid = self.get_midprice()
        book = self.asks if side == "BUY" else self.bids

        fills: list[tuple[float, int]] = []
        remaining = order_size
        levels_swept = 0

        for i, (px, sz) in enumerate(book):
            if remaining <= 0:
                break
            filled = min(remaining, sz)
            fills.append((px, filled))
            remaining -= filled
            if filled == sz:
                levels_swept += 1
                book[i] = (px, 0)
            else:
                book[i] = (px, sz - filled)

        # Remove fully swept levels from the front.
        del book[:levels_swept]

        total_filled = order_size - remaining
        vwap = (
            sum(px * qty for px, qty in fills) / total_filled
            if total_filled > 0
            else 0.0
        )

        if mid is not None and total_filled > 0:
            arrival_cost = vwap - mid if side == "BUY" else mid - vwap
        else:
            arrival_cost = None

        self._sync_book_to_arrays()
        self._apply_impact_to_future(side, fills)

        return {
            "vwap": vwap,
            "fills": fills,
            "total_filled": total_filled,
            "levels_swept": levels_swept,
            "executed_at_ns": int(self.ts_ns[self._cur_idx]),
            "executed_at_idx": self._cur_idx,
            "arrival_cost": arrival_cost,
        }

    def accept_ioc_lmt_order(self, side: str, limit_price: float, order_size: int, ts_nano: int, latency_ns: int = 0) -> dict:
        """
        Simulate an IOC Limit order execution at a given timestamp.

        Parameters
        ----------
        side : str
            ``"BUY"`` (walks asks) or ``"SELL"`` (walks bids).
        limit_price: float
            Limit price.
        order_size : int
            Number of shares to fill.
        ts_nano : int
            Nanosecond timestamp at which the order is sent.
        latency_ns : int
            Additional latency in nanoseconds before execution (default 0).

        Returns
        -------
        dict
            ``vwap`` — volume-weighted average fill price.
            ``fills`` — list of ``(price, qty_filled)`` tuples.
            ``total_filled`` — total shares filled (may be < *order_size*).
            ``levels_swept`` — number of levels fully consumed.
            ``executed_at_ns`` — ``ts_event_nano`` of the book row the order
            executed against.
            ``executed_at_idx`` — integer index of that row in the internal
            arrays (useful for slicing).
            ``arrival_cost`` — difference between VWAP and midprice at
            execution time (positive means paid more than mid for BUY,
            received less than mid for SELL).
        """
        self.advance_to(ts_nano + latency_ns)
        mid = self.get_midprice()
        book = self.asks if side == "BUY" else self.bids

        fills: list[tuple[float, int]] = []
        remaining = order_size
        levels_swept = 0

        for i, (px, sz) in enumerate(book):
            if remaining <= 0:
                break
            if px > limit_price and side == "BUY":
                break
            if px < limit_price and side == "SELL":
                break
            filled = min(remaining, sz)
            fills.append((px, filled))
            remaining -= filled
            if filled == sz:
                levels_swept += 1
                book[i] = (px, 0)
            else:
                book[i] = (px, sz - filled)

        # Remove fully swept levels from the front.
        del book[:levels_swept]

        total_filled = order_size - remaining
        vwap = (
            sum(px * qty for px, qty in fills) / total_filled
            if total_filled > 0
            else 0.0
        )

        if mid is not None and total_filled > 0:
            arrival_cost = vwap - mid if side == "BUY" else mid - vwap
        else:
            arrival_cost = None

        self._sync_book_to_arrays()
        self._apply_impact_to_future(side, fills)

        return {
            "vwap": vwap,
            "fills": fills,
            "total_filled": total_filled,
            "levels_swept": levels_swept,
            "executed_at_ns": int(self.ts_ns[self._cur_idx]),
            "executed_at_idx": self._cur_idx,
            "arrival_cost": arrival_cost,
        }

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def get_book_state(self) -> dict:
        """
        Return the current internal book state.

        Returns
        -------
        dict
            ``bids`` — list of ``(price, size)`` tuples.
            ``asks`` — list of ``(price, size)`` tuples.
            ``ts_nano`` — nanosecond timestamp of the current snapshot.
        """
        return {
            "bids": list(self.bids),
            "asks": list(self.asks),
            "ts_nano": int(self.ts_ns[self._cur_idx]),
        }

    def get_midprice(self) -> float | None:
        """
        Return the midprice (average of best bid and best ask).

        Returns
        -------
        float or None
            ``None`` if either side of the book is empty.
        """
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2.0

    def get_book_df(self) -> pl.DataFrame:
        """
        Return the full orderbook across all timestamps as a Polars DataFrame.

        Values are read from the internal numpy arrays, so they reflect any
        post-execution modifications propagated to future rows.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``ts_event_nano``, ``instrument_id``, and 10 levels
            of ``bid_px_XX``, ``ask_px_XX``, ``bid_sz_XX``, ``ask_sz_XX``.
        """
        data: dict[str, np.ndarray] = {
            "ts_event_nano": self.ts_ns,
            "instrument_id": self.instrument_id,
        }
        for l in range(self._N_LEVELS):
            tag = f"{l:02d}"
            data[f"bid_px_{tag}"] = self.bid_px[:, l]
            data[f"ask_px_{tag}"] = self.ask_px[:, l]
            data[f"bid_sz_{tag}"] = self.bid_sz[:, l]
            data[f"ask_sz_{tag}"] = self.ask_sz[:, l]
        return pl.DataFrame(data)

