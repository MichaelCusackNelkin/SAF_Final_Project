"""
Order book simulator
"""

import numpy as np
import pandas as pd
import databento as db
import polars as pl

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
    replenishment_rate : float
        Lambda parameter for the Poisson process that draws new liquidity
        sizes when replenishing swept price levels after an execution.
        A value of 0 disables replenishment (default).
    latency_ns : int
        Simulated latency in nanoseconds added to every order's timestamp
        before execution (default 0).
    """

    _N_LEVELS: int = 10

    def __init__(self, df: pl.DataFrame, replenishment_rate: float = 0.0, latency_ns: int = 0) -> None:
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
        self.replenishment_rate: float = replenishment_rate
        self.latency_ns: int = latency_ns

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
    
    def _compute_replenishment(self, side: str) -> list[tuple[float, int]]:
        """
        Draw Poisson-distributed liquidity for swept gap levels after execution.

        This method only *computes* which levels to add and their sizes — it
        does **not** modify ``self.bids`` / ``self.asks``.  The replenished
        liquidity is written exclusively to future rows (starting at
        ``_cur_idx + 1``) by ``_apply_replenishment_to_future``, so the fill
        row itself reflects the depleted book state.

        For a BUY order (asks were consumed): replenish ask levels from the
        current best ask down to ``best_bid + 0.01``.

        For a SELL order (bids were consumed): replenish bid levels from the
        current best bid up to ``best_ask - 0.01``.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"`` — the side of the *order* that executed.

        Returns
        -------
        list[tuple[float, int]]
            Levels to be inserted, as ``(price, size)`` tuples.
        """
        if self.replenishment_rate <= 0:
            return []

        added: list[tuple[float, int]] = []

        if side == "BUY":
            if not self.bids or not self.asks:
                return []
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            existing_ask_prices = {round(px, 2) for px, _ in self.asks}
            level_px = round(best_ask - 0.01, 2)
            floor_px = round(best_bid + 0.01, 2)
            while level_px >= floor_px:
                if level_px not in existing_ask_prices:
                    sz = int(np.random.poisson(self.replenishment_rate))
                    if sz > 0:
                        added.append((level_px, sz))
                level_px = round(level_px - 0.01, 2)
        else:
            if not self.bids or not self.asks:
                return []
            best_ask = self.asks[0][0]
            best_bid = self.bids[0][0]
            existing_bid_prices = {round(px, 2) for px, _ in self.bids}
            level_px = round(best_bid + 0.01, 2)
            ceiling_px = round(best_ask - 0.01, 2)
            while level_px <= ceiling_px:
                if level_px not in existing_bid_prices:
                    sz = int(np.random.poisson(self.replenishment_rate))
                    if sz > 0:
                        added.append((level_px, sz))
                level_px = round(level_px + 0.01, 2)

        return added

    def _apply_replenishment_to_future(self, side: str, added: list[tuple[float, int]]) -> None:
        """
        Propagate replenished liquidity into future rows starting at
        ``_cur_idx + 1`` (the next timestamp after the fill).

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"`` — the side of the *order* that executed.
            A BUY replenishes asks; a SELL replenishes bids.
        added : list[tuple[float, int]]
            List of ``(price, size)`` tuples to add.
        """
        start = self._cur_idx + 1
        if start >= self._n_rows or not added:
            return

        if side == "BUY":
            px_arr = self.ask_px
            sz_arr = self.ask_sz
        else:
            px_arr = self.bid_px
            sz_arr = self.bid_sz

        future_px = px_arr[start:]
        future_sz = sz_arr[start:]

        for add_px, add_qty in added:
            mask = future_px == add_px
            future_sz[mask] += add_qty

    def _sync_book_to_arrays(self) -> None:
        """Write ``self.bids`` / ``self.asks`` back into the numpy arrays at
        ``self._cur_idx``, shifting remaining levels to the front and
        back-filling any trailing empty slots with synthetic liquidity
        (one tick worse per level, random size 100-150)."""
        idx = self._cur_idx

        # --- bids ---
        n_bids = len(self.bids)
        for l in range(self._N_LEVELS):
            if l < n_bids:
                self.bid_px[idx, l] = self.bids[l][0]
                self.bid_sz[idx, l] = self.bids[l][1]
            else:
                prev_px = self.bid_px[idx, l - 1]
                syn_px = round(prev_px - 0.01, 2)
                self.bid_px[idx, l] = syn_px
                self.bid_sz[idx, l] = 100
                self.bids.append((syn_px, 100))

        # --- asks ---
        n_asks = len(self.asks)
        for l in range(self._N_LEVELS):
            if l < n_asks:
                self.ask_px[idx, l] = self.asks[l][0]
                self.ask_sz[idx, l] = self.asks[l][1]
            else:
                prev_px = self.ask_px[idx, l - 1]
                syn_px = round(prev_px + 0.01, 2)
                self.ask_px[idx, l] = syn_px
                self.ask_sz[idx, l] = 100
                self.asks.append((syn_px, 100))

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
            for l in range(n_keep, self._N_LEVELS):
                prev = future_px[row, l - 1]
                if side == "BUY":
                    future_px[row, l] = round(prev + 0.01, 2)
                else:
                    future_px[row, l] = round(prev - 0.01, 2)
                future_sz[row, l] = 100

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

    def accept_mkt_order(self, side: str, order_size: int, ts_nano: int) -> dict:
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
        self.advance_to(ts_nano + self.latency_ns)
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

        replenished = self._compute_replenishment(side)
        self._sync_book_to_arrays()
        self._apply_impact_to_future(side, fills)
        self._apply_replenishment_to_future(side, replenished)

        return {
            "vwap": vwap,
            "fills": fills,
            "total_filled": total_filled,
            "levels_swept": levels_swept,
            "executed_at_ns": int(self.ts_ns[self._cur_idx]),
            "executed_at_idx": self._cur_idx,
            "arrival_cost": arrival_cost,
        }

    def accept_ioc_lmt_order(self, side: str, limit_price: float, order_size: int, ts_nano: int) -> dict:
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
        self.advance_to(ts_nano + self.latency_ns)
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

        replenished = self._compute_replenishment(side)
        self._sync_book_to_arrays()
        self._apply_impact_to_future(side, fills)
        self._apply_replenishment_to_future(side, replenished)

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
    # filling strategies
    # ------------------------------------------------------------------

    def naive_fill(
        self, side: str, parent_size: int, ts_nano: int
    ) -> tuple[dict, pl.DataFrame]:
        """
        Execute a single market order for the full parent size.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"``.
        parent_size : int
            Total number of shares to fill.
        ts_nano : int
            Nanosecond timestamp at which the order is sent.

        Returns
        -------
        tuple[dict, pl.DataFrame]
            The execution result dict and the post-execution book DataFrame.
        """
        result = self.accept_mkt_order(side, parent_size, ts_nano)
        return result, self.get_book_df()

    def constant_fill(
        self,
        side: str,
        parent_size: int,
        start_ts_ns: int,
        wait_ns: int,
        n_chunks: int,
    ) -> tuple[list[dict], pl.DataFrame]:
        """
        Split a parent order into equal-sized market-order chunks.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"``.
        parent_size : int
            Total number of shares to fill.
        start_ts_ns : int
            Nanosecond timestamp of the first child order.
        wait_ns : int
            Nanosecond interval between child orders.
        n_chunks : int
            Number of child orders.

        Returns
        -------
        tuple[list[dict], pl.DataFrame]
            List of execution result dicts and the post-execution book
            DataFrame.
        """
        chunk_size = int(parent_size / n_chunks)
        timestamps = np.arange(start_ts_ns, start_ts_ns + n_chunks * wait_ns, wait_ns)
        results: list[dict] = []
        for t in timestamps:
            r = self.accept_mkt_order(side, chunk_size, int(t))
            results.append(r)
        return results, self.get_book_df()

    def constant_ioc_fill(
        self,
        side: str,
        limit_px: float,
        parent_size: int,
        start_ts_ns: int,
        wait_ns: int,
        n_chunks: int,
    ) -> tuple[list[dict], pl.DataFrame]:
        """
        Split a parent order into equal-sized IOC limit-order chunks.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"``.
        limit_px : float
            Limit price for every child IOC order.
        parent_size : int
            Total number of shares to fill.
        start_ts_ns : int
            Nanosecond timestamp of the first child order.
        wait_ns : int
            Nanosecond interval between child orders.
        n_chunks : int
            Number of child orders.

        Returns
        -------
        tuple[list[dict], pl.DataFrame]
            List of execution result dicts and the post-execution book
            DataFrame.
        """
        chunk_size = int(parent_size / n_chunks)
        timestamps = np.arange(start_ts_ns, start_ts_ns + n_chunks * wait_ns, wait_ns)
        results: list[dict] = []
        for t in timestamps:
            r = self.accept_ioc_lmt_order(side, limit_px, chunk_size, int(t))
            results.append(r)
        return results, self.get_book_df()

    def obi_adaptive_fill(
        self,
        side: str,
        parent_size: int,
        start_ts_ns: int,
        max_size: int,
        threshold: float,
        horizon: int,
        limit_price: float | None = None,
    ) -> tuple[list[dict], pl.DataFrame, pl.DataFrame]:
        """
        Adaptively fill a parent order using order-book imbalance (OBI).

        At each step the method scans the book for timestamps within the
        rolling *horizon* window, computes OBI, derives order sizes
        proportional to ``|OBI| * max_size``, and sends IOC limit orders when
        the imbalance exceeds *threshold* in the favourable direction.

        Parameters
        ----------
        side : str
            ``"BUY"`` or ``"SELL"``.
        parent_size : int
            Total number of shares to fill.
        max_size : int
            Maximum child-order size (scaled by ``|OBI|``).
        threshold : float
            OBI threshold that must be exceeded before a child order fires.
        start_ts_ns : int
            Nanosecond timestamp at which to begin scanning.
        horizon : int
            Nanosecond look-ahead window for candidate selection.
        limit_price : float or None, optional
            Fixed limit price for all child orders.  When ``None`` (default),
            the limit is computed dynamically from each candidate row using
            ``ask_px_02`` (BUY) or ``bid_px_02`` (SELL).

        Returns
        -------
        tuple[list[dict], pl.DataFrame, pl.DataFrame]
            ``(results, final_book_df, last_candidates)`` — the list of
            execution result dicts, the post-execution book DataFrame, and the
            candidates DataFrame from the last iteration.
        """
        init_ts_ns = start_ts_ns
        results: list[dict] = []
        remaining = parent_size
        book_df = self.get_book_df()
        candidates = pl.DataFrame()

        while remaining > 0 and start_ts_ns < horizon + init_ts_ns:
            candidates = (
                book_df
                .filter(
                    pl.col("ts_event_nano").is_between(start_ts_ns, start_ts_ns + horizon)
                )
                .with_columns(
                    pl.col("bid_sz_00").fill_null(0).fill_nan(0).alias("bid_sz_00"),
                    pl.col("ask_sz_00").fill_null(0).fill_nan(0).alias("ask_sz_00"),
                )
                .with_columns(
                    (
                        (pl.col("bid_sz_00").cast(pl.Int32) - pl.col("ask_sz_00").cast(pl.Int32))
                        / (pl.col("bid_sz_00").cast(pl.Int32) + pl.col("ask_sz_00").cast(pl.Int32))
                    ).alias("obi")
                )
                .with_columns(
                    (pl.col("obi").abs() * max_size)
                    .clip(1, None)
                    .cast(pl.Int64)
                    .alias("order_size")
                )
            )

            if side == "BUY":
                candidates = candidates.filter(pl.col("obi") > threshold)
            else:
                candidates = candidates.filter(pl.col("obi") < threshold)

            if candidates.is_empty():
                break

            row = candidates.row(0, named=True)
            size = min(row["order_size"], remaining)

            if size <= 0:
                break

            if limit_price is not None:
                limit = limit_price
            else:
                limit = row["ask_px_02"] if side == "BUY" else row["bid_px_02"]

            r = self.accept_ioc_lmt_order(side, limit, size, row["ts_event_nano"])
            remaining -= r["total_filled"]
            results.append(r)

            start_ts_ns = row["ts_event_nano"] + 1
            book_df = self.get_book_df()

        return results, self.get_book_df(), candidates

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

