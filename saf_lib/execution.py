"""
execution.py — Market microstructure analysis module for measuring
alpha decay vs market impact when splitting orders.

Part of the SAF (Statistical Aspects of Finance) final project.
Data source: DataBentoMBP10Adapter, ~5.7M rows of MBP-10 tick data
(AMZN, XNAS.ITCH).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

__all__ = [
    "compute_features",
    "compute_markouts",
    "markout_by_signal",
    "ExecutionSimulator",
    "compare_strategies",
    "plot_markout_curves",
    "plot_strategy_comparison",
    "plot_book_imbalance_vs_markout",
    "plot_impact_vs_alpha_decay",
    "plot_spread_and_imbalance_timeseries",
]


# ---------------------------------------------------------------------------
# 1. Data Ingestion / Feature Engineering
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add microstructure feature columns to the raw MBP-10 DataFrame.

    New columns added:
        mid_price, spread, spread_bps, micro_price, trade_flow,
        cumulative_trade_flow, obi_level0, volume_imbalance.

    Parameters
    ----------
    df : pd.DataFrame
        Raw MBP-10 DataFrame from DataBentoMBP10Adapter.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with new columns appended (existing columns
        are preserved).
    """
    df = df.copy()

    # Mid-price and spread
    df["mid_price"] = (df["bid_px_00"] + df["ask_px_00"]) / 2.0
    df["spread"] = df["ask_px_00"] - df["bid_px_00"]
    df["spread_bps"] = df["spread"] / df["mid_price"] * 10_000

    # Micro-price (size-weighted mid)
    total_sz = df["bid_sz_00"] + df["ask_sz_00"]
    df["micro_price"] = np.where(
        total_sz > 0,
        (df["bid_px_00"] * df["ask_sz_00"] + df["ask_px_00"] * df["bid_sz_00"]) / total_sz,
        df["mid_price"],
    )

    # Signed trade flow
    is_trade = df["action"] == "T"
    buyer = is_trade & (df["price"] >= df["ask_px_00"])
    seller = is_trade & (df["price"] <= df["bid_px_00"])
    df["trade_flow"] = 0.0
    df.loc[buyer, "trade_flow"] = df.loc[buyer, "size"].astype(float)
    df.loc[seller, "trade_flow"] = -df.loc[seller, "size"].astype(float)

    df["cumulative_trade_flow"] = df["trade_flow"].cumsum()

    # Top-of-book order-book imbalance
    bid_ask_sum = df["bid_sz_00"] + df["ask_sz_00"]
    df["obi_level0"] = np.where(
        bid_ask_sum > 0,
        (df["bid_sz_00"].astype(np.int64) - df["ask_sz_00"].astype(np.int64)) / bid_ask_sum,
        0.0,
    )

    # Rolling volume imbalance (1000-tick window)
    rolling_flow = df["trade_flow"].rolling(1000, min_periods=1).sum()
    rolling_abs = df["trade_flow"].abs().rolling(1000, min_periods=1).sum()
    df["volume_imbalance"] = np.where(
        rolling_abs > 0,
        rolling_flow / rolling_abs,
        np.nan,
    )

    return df


# ---------------------------------------------------------------------------
# 2. Markout Analysis
# ---------------------------------------------------------------------------

_DEFAULT_HORIZONS_MS = [100, 500, 1000, 5000, 10_000, 30_000, 60_000]


def compute_markouts(
    df: pd.DataFrame,
    horizons_ms: Optional[list[int]] = None,
    sample_n: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute mid-price markouts at multiple future horizons.

    For each sampled row, the markout at horizon *h* is the mid-price
    change from the current tick to the first tick whose ``ts_recv``
    is >= current ``ts_recv`` + *h* milliseconds.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame (must contain ``mid_price``).
    horizons_ms : list[int], optional
        Future horizons in milliseconds. Defaults to
        [100, 500, 1000, 5000, 10000, 30000, 60000].
    sample_n : int
        Number of rows to sample (default 50 000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per sampled tick with markout columns.
    """
    if horizons_ms is None:
        horizons_ms = _DEFAULT_HORIZONS_MS

    # Work with integer nanosecond timestamps for speed
    ts_ns = df.index.astype(np.int64).values  # nanosecond epoch
    mid = df["mid_price"].values

    n = len(df)
    rng = np.random.default_rng(seed)
    sample_idx = np.sort(rng.choice(n, size=min(sample_n, n), replace=False))

    records = {
        "ts_recv": df.index[sample_idx],
        "mid_price": mid[sample_idx],
    }

    # Carry over useful signal columns if present
    for col in ("book_imbalance", "obi_level0", "spread_bps"):
        if col in df.columns:
            records[col] = df[col].values[sample_idx]

    for h in horizons_ms:
        h_ns = int(h) * 1_000_000  # ms -> ns
        future_ts = ts_ns[sample_idx] + h_ns
        future_idx = np.searchsorted(ts_ns, future_ts, side="left")
        # Clamp to valid range
        future_idx = np.clip(future_idx, 0, n - 1)
        future_mid = mid[future_idx]
        current_mid = mid[sample_idx]

        markout = future_mid - current_mid
        markout_bps = np.where(
            current_mid > 0, markout / current_mid * 10_000, np.nan
        )

        records[f"markout_{h}ms"] = markout
        records[f"markout_bps_{h}ms"] = markout_bps

    return pd.DataFrame(records)


def markout_by_signal(
    markouts_df: pd.DataFrame,
    signal_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin rows by a signal column and compute mean markouts per bin.

    Parameters
    ----------
    markouts_df : pd.DataFrame
        Output of :func:`compute_markouts`.
    signal_col : str
        Column to bin on (e.g. ``"book_imbalance"``, ``"obi_level0"``).
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    pd.DataFrame
        Indexed by bin label, with one column per horizon (bps markout).
    """
    tmp = markouts_df.copy()
    tmp["bin"] = pd.qcut(tmp[signal_col], n_bins, duplicates="drop")

    markout_cols = [c for c in tmp.columns if c.startswith("markout_bps_")]
    result = tmp.groupby("bin", observed=True)[markout_cols].mean()
    return result


# ---------------------------------------------------------------------------
# 3. Execution Strategies
# ---------------------------------------------------------------------------

class ExecutionSimulator:
    """Simulate execution of a parent buy order using different strategies.

    The simulator operates on a feature-enriched DataFrame and evaluates
    execution quality via arrival cost in basis points.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame (output of :func:`compute_features`).
    parent_qty : int
        Total shares to purchase (default 10 000).
    horizon_seconds : float
        Execution horizon in seconds (default 300 = 5 minutes).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        parent_qty: int = 10_000,
        horizon_seconds: float = 300.0,
    ):
        self.df = df
        self.parent_qty = parent_qty
        self.horizon_seconds = horizon_seconds

        # Pre-compute convenience arrays
        self.ts_ns = df.index.astype(np.int64).values
        self.horizon_ns = int(horizon_seconds * 1e9)

    # ---- internal helpers ------------------------------------------------

    def _window(self, start_idx: int) -> pd.DataFrame:
        """Return the sub-DataFrame within the execution horizon."""
        start_ns = self.ts_ns[start_idx]
        end_ns = start_ns + self.horizon_ns
        end_idx = np.searchsorted(self.ts_ns, end_ns, side="right")
        return self.df.iloc[start_idx:end_idx]

    @staticmethod
    def _result(fills: list[dict], arrival_mid: float, parent_qty: int) -> dict:
        """Aggregate a list of fills into a result summary."""
        if not fills:
            return {
                "avg_price": np.nan,
                "arrival_cost_bps": np.nan,
                "total_shares_filled": 0,
                "fill_rate": 0.0,
                "fills": fills,
            }
        total_qty = sum(f["qty"] for f in fills)
        avg_px = sum(f["qty"] * f["price"] for f in fills) / total_qty if total_qty > 0 else np.nan
        cost_bps = (avg_px - arrival_mid) / arrival_mid * 10_000 if arrival_mid > 0 else np.nan
        return {
            "avg_price": avg_px,
            "arrival_cost_bps": cost_bps,
            "total_shares_filled": total_qty,
            "fill_rate": total_qty / parent_qty,
            "fills": fills,
        }

    # ---- strategies ------------------------------------------------------

    def twap(self, n_slices: int = 20, start_idx: int = 0) -> dict:
        """Time-Weighted Average Price execution.

        Splits the horizon into *n_slices* equal intervals and executes
        ``parent_qty / n_slices`` shares at the prevailing ask at the
        start of each interval.

        Parameters
        ----------
        n_slices : int
            Number of time slices.
        start_idx : int
            Index into ``self.df`` at which execution begins.

        Returns
        -------
        dict
            Execution result with avg_price, arrival_cost_bps, etc.
        """
        window = self._window(start_idx)
        if window.empty:
            return self._result([], np.nan, self.parent_qty)

        arrival_mid = window["mid_price"].iloc[0]
        ts_ns_w = window.index.astype(np.int64).values
        start_ns = ts_ns_w[0]
        slice_ns = self.horizon_ns / n_slices
        qty_per_slice = self.parent_qty / n_slices

        fills: list[dict] = []
        remaining = self.parent_qty

        for i in range(n_slices):
            if remaining <= 0:
                break
            target_ns = start_ns + int(i * slice_ns)
            loc = np.searchsorted(ts_ns_w, target_ns, side="left")
            if loc >= len(window):
                break
            row = window.iloc[loc]
            qty = min(qty_per_slice, remaining)
            fills.append({"ts": window.index[loc], "price": row["ask_px_00"], "qty": qty})
            remaining -= qty

        return self._result(fills, arrival_mid, self.parent_qty)

    def vwap(self, n_slices: int = 20, start_idx: int = 0) -> dict:
        """Volume-Weighted Average Price execution.

        Splits the horizon into *n_slices* intervals and allocates
        shares proportionally to observed trade volume in each interval.

        Parameters
        ----------
        n_slices : int
            Number of time slices.
        start_idx : int
            Index into ``self.df`` at which execution begins.

        Returns
        -------
        dict
            Execution result.
        """
        window = self._window(start_idx)
        if window.empty:
            return self._result([], np.nan, self.parent_qty)

        arrival_mid = window["mid_price"].iloc[0]
        ts_ns_w = window.index.astype(np.int64).values
        start_ns = ts_ns_w[0]
        slice_ns = self.horizon_ns / n_slices

        # Compute volume per slice
        trades = window[window["action"] == "T"]
        slice_volumes = np.zeros(n_slices)
        if not trades.empty:
            trade_ts = trades.index.astype(np.int64).values
            trade_sizes = trades["size"].values.astype(float)
            slice_indices = np.clip(
                ((trade_ts - start_ns) / slice_ns).astype(int), 0, n_slices - 1
            )
            np.add.at(slice_volumes, slice_indices, trade_sizes)

        total_volume = slice_volumes.sum()
        if total_volume == 0:
            # Fall back to uniform
            weights = np.ones(n_slices) / n_slices
        else:
            weights = slice_volumes / total_volume

        fills: list[dict] = []
        remaining = self.parent_qty

        for i in range(n_slices):
            if remaining <= 0:
                break
            target_ns = start_ns + int(i * slice_ns)
            loc = np.searchsorted(ts_ns_w, target_ns, side="left")
            if loc >= len(window):
                break
            row = window.iloc[loc]
            qty = min(round(weights[i] * self.parent_qty), remaining)
            if qty <= 0:
                continue
            fills.append({"ts": window.index[loc], "price": row["ask_px_00"], "qty": qty})
            remaining -= qty

        # Allocate any rounding remainder to the last slice
        if remaining > 0 and fills:
            fills[-1]["qty"] += remaining

        return self._result(fills, arrival_mid, self.parent_qty)

    def obi_adaptive(
        self,
        n_slices: int = 20,
        aggression_threshold: float = 0.3,
        start_idx: int = 0,
    ) -> dict:
        """Order Book Imbalance adaptive execution.

        Uses ``book_imbalance`` to decide aggressiveness in each slice:
        - OBI > threshold  -> aggressive (buy at ask immediately)
        - OBI < -threshold -> passive (limit at bid; fills only if a
          trade prints at or below the bid in that interval)
        - otherwise        -> execute at mid

        Parameters
        ----------
        n_slices : int
            Number of time slices.
        aggression_threshold : float
            OBI threshold for switching between aggressive / passive.
        start_idx : int
            Index into ``self.df`` at which execution begins.

        Returns
        -------
        dict
            Execution result.
        """
        window = self._window(start_idx)
        if window.empty:
            return self._result([], np.nan, self.parent_qty)

        arrival_mid = window["mid_price"].iloc[0]
        ts_ns_w = window.index.astype(np.int64).values
        start_ns = ts_ns_w[0]
        slice_ns = self.horizon_ns / n_slices
        qty_per_slice = self.parent_qty / n_slices

        fills: list[dict] = []
        remaining = self.parent_qty

        for i in range(n_slices):
            if remaining <= 0:
                break
            slice_start_ns = start_ns + int(i * slice_ns)
            slice_end_ns = start_ns + int((i + 1) * slice_ns)

            loc_start = np.searchsorted(ts_ns_w, slice_start_ns, side="left")
            loc_end = np.searchsorted(ts_ns_w, slice_end_ns, side="left")
            if loc_start >= len(window):
                break

            row = window.iloc[loc_start]
            obi = row.get("book_imbalance", 0.0) #FIXME
            qty = min(qty_per_slice, remaining)

            if obi > aggression_threshold:
                # Aggressive: buy at ask
                fills.append({
                    "ts": window.index[loc_start],
                    "price": row["ask_px_00"],
                    "qty": qty,
                })
                remaining -= qty

            elif obi < -aggression_threshold:
                # Passive: limit at bid — check if any trade in the
                # interval printed at or below our bid price
                bid_px = row["bid_px_00"]
                slice_df = window.iloc[loc_start:loc_end]
                slice_trades = slice_df[
                    (slice_df["action"] == "T") & (slice_df["price"] <= bid_px)
                ]
                if not slice_trades.empty:
                    fills.append({
                        "ts": slice_trades.index[0],
                        "price": bid_px,
                        "qty": qty,
                    })
                    remaining -= qty
                # else: no fill this slice (passive order not executed)

            else:
                # Neutral: execute at mid
                fills.append({
                    "ts": window.index[loc_start],
                    "price": row["mid_price"],
                    "qty": qty,
                })
                remaining -= qty

        return self._result(fills, arrival_mid, self.parent_qty)

    def immediate(self, start_idx: int = 0) -> dict:
        """Execute the entire order immediately at the ask.

        Maximum market impact, zero alpha decay. Serves as a baseline.

        Parameters
        ----------
        start_idx : int
            Index into ``self.df`` at which execution begins.

        Returns
        -------
        dict
            Execution result.
        """
        if start_idx >= len(self.df):
            return self._result([], np.nan, self.parent_qty)

        row = self.df.iloc[start_idx]
        arrival_mid = row["mid_price"]
        fills = [{"ts": self.df.index[start_idx], "price": row["ask_px_00"], "qty": self.parent_qty}]
        return self._result(fills, arrival_mid, self.parent_qty)


def compare_strategies(
    simulator: ExecutionSimulator,
    n_simulations: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Run each execution strategy from random starting points.

    Parameters
    ----------
    simulator : ExecutionSimulator
        A configured simulator instance.
    n_simulations : int
        Number of random starting points.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: strategy, arrival_cost_bps, fill_rate.
        One row per simulation per strategy.
    """
    rng = np.random.default_rng(seed)
    n = len(simulator.df)
    # Ensure there is enough room for the full execution horizon
    max_start = np.searchsorted(
        simulator.ts_ns,
        simulator.ts_ns[-1] - simulator.horizon_ns,
        side="right",
    )
    max_start = max(1, min(max_start, n - 1))
    start_indices = rng.choice(max_start, size=n_simulations, replace=True)

    strategies = {
        "immediate": lambda idx: simulator.immediate(start_idx=idx),
        "twap": lambda idx: simulator.twap(start_idx=idx),
        "vwap": lambda idx: simulator.vwap(start_idx=idx),
        "obi_adaptive": lambda idx: simulator.obi_adaptive(start_idx=idx),
    }

    rows: list[dict] = []
    for idx in start_indices:
        for name, fn in strategies.items():
            res = fn(int(idx))
            rows.append({
                "strategy": name,
                "arrival_cost_bps": res["arrival_cost_bps"],
                "fill_rate": res["fill_rate"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Plotting Functions
# ---------------------------------------------------------------------------

def plot_markout_curves(
    markouts_df: pd.DataFrame,
    signal_col: str,
    n_bins: int = 5,
) -> plt.Figure:
    """Plot average markout (bps) vs horizon for quantile bins of a signal.

    Parameters
    ----------
    markouts_df : pd.DataFrame
        Output of :func:`compute_markouts`.
    signal_col : str
        Column to bin on.
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    matplotlib.figure.Figure
    """
    binned = markout_by_signal(markouts_df, signal_col, n_bins=n_bins)
    horizons = sorted(
        int(c.replace("markout_bps_", "").replace("ms", ""))
        for c in binned.columns
    )
    cols = [f"markout_bps_{h}ms" for h in horizons]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.RdYlGn
    for i, (label, row) in enumerate(binned.iterrows()):
        color = cmap(i / max(len(binned) - 1, 1))
        ax.plot(horizons, row[cols].values, marker="o", label=str(label), color=color)

    ax.set_xlabel("Horizon (ms)")
    ax.set_ylabel("Markout (bps)")
    ax.set_title(f"Markout Curves by {signal_col}")
    ax.legend(title=signal_col, fontsize=8, loc="best")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_strategy_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Box plot of arrival cost (bps) for each execution strategy.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of :func:`compare_strategies`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    strategies = results_df["strategy"].unique()
    data = [
        results_df.loc[results_df["strategy"] == s, "arrival_cost_bps"].dropna().values
        for s in strategies
    ]
    bp = ax.boxplot(data, labels=strategies, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Arrival Cost (bps)")
    ax.set_title("Execution Strategy Cost Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_book_imbalance_vs_markout(
    markouts_df: pd.DataFrame,
    horizon_ms: int = 1000,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Scatter / binned-line plot of book imbalance vs markout.

    Parameters
    ----------
    markouts_df : pd.DataFrame
        Output of :func:`compute_markouts`.
    horizon_ms : int
        Horizon to use for the markout column.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If *None*, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    col = f"markout_bps_{horizon_ms}ms"
    tmp = markouts_df[["book_imbalance", col]].dropna()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Binned means for cleaner visualization
    tmp["bin"] = pd.qcut(tmp["book_imbalance"], 50, duplicates="drop")
    binned = tmp.groupby("bin", observed=True).agg(
        x=("book_imbalance", "mean"),
        y=(col, "mean"),
    )
    ax.scatter(binned["x"], binned["y"], alpha=0.7, s=30, zorder=3)

    # Linear fit
    x, y = binned["x"].values, binned["y"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 1:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        fit_x = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(fit_x, np.polyval(coeffs, fit_x), "r--", linewidth=2,
                label=f"fit: {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
        ax.legend()

    ax.set_xlabel("Book Imbalance")
    ax.set_ylabel(f"Markout (bps) @ {horizon_ms}ms")
    ax.set_title(f"Book Imbalance vs {horizon_ms}ms Markout")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_impact_vs_alpha_decay(results_df: pd.DataFrame) -> plt.Figure:
    """Plot fill rate vs arrival cost for each strategy.

    Visualizes the market-impact / alpha-decay tradeoff.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of :func:`compare_strategies`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    summary = results_df.groupby("strategy").agg(
        fill_rate=("fill_rate", "mean"),
        arrival_cost_bps=("arrival_cost_bps", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(summary)))
    for (name, row), color in zip(summary.iterrows(), colors):
        ax.scatter(row["fill_rate"], row["arrival_cost_bps"], s=120, color=color, zorder=3)
        ax.annotate(
            name,
            (row["fill_rate"], row["arrival_cost_bps"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Fill Rate")
    ax.set_ylabel("Arrival Cost (bps)")
    ax.set_title("Arrival Cost vs Fill Rate (means)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spread_and_imbalance_timeseries(
    df: pd.DataFrame,
    start_time: Optional[str] = None,
    duration_minutes: float = 5.0,
) -> plt.Figure:
    """Two-panel plot of spread (bps) and book imbalance over a short window.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame.
    start_time : str, optional
        Starting timestamp (parseable by pandas). If *None*, a random
        5-minute window is selected.
    duration_minutes : float
        Window length in minutes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    duration_ns = int(duration_minutes * 60 * 1e9)
    ts_ns = df.index.astype(np.int64).values

    if start_time is None:
        max_start = np.searchsorted(ts_ns, ts_ns[-1] - duration_ns, side="right")
        max_start = max(1, max_start)
        rng = np.random.default_rng()
        start_idx = rng.integers(0, max_start)
    else:
        start_ts = pd.Timestamp(start_time)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("US/Eastern")
        start_idx = np.searchsorted(ts_ns, start_ts.value, side="left")

    end_ns = ts_ns[start_idx] + duration_ns
    end_idx = np.searchsorted(ts_ns, end_ns, side="right")
    window = df.iloc[start_idx:end_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(window.index, window["spread_bps"], linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("Spread (bps)")
    ax1.set_title("Spread and Book Imbalance")
    ax1.grid(True, alpha=0.3)

    ax2.plot(window.index, window["book_imbalance"], linewidth=0.5, alpha=0.8, color="tab:orange")
    ax2.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Book Imbalance")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
