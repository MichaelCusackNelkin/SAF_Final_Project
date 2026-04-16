"""
execution.py — Mid-price markout analysis for MBP-10 tick data.

Exposes two functions:
    - compute_markouts: sample ticks and measure forward mid-price moves
      at configurable horizons (nanoseconds, so μs → s is supported).
    - plot_markout_curves: plot mean markout (bps) vs horizon, binned by
      a signal column (``book_imbalance`` or ``obi_level0``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from typing import Optional, Sequence

__all__ = ["compute_markouts", "plot_markout_curves"]


# One microsecond through ten seconds, log-spaced.
_DEFAULT_HORIZONS_NS: list[int] = [
    1_000, 10_000, 100_000,
    1_000_000, 10_000_000, 100_000_000,
    1_000_000_000, 10_000_000_000,
]

_LEVELS = range(10)


def compute_markouts(
    df: pl.DataFrame,
    horizons_ns: Optional[Sequence[int]] = None,
    sample_n: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute forward mid-price markouts at multiple horizons.

    For each sampled tick the markout at horizon *h* (ns) is the
    mid-price change from the current tick to the first tick whose
    timestamp is ≥ current_ts + *h*.

    Parameters
    ----------
    df : pl.DataFrame
        MBP-10 polars frame. Must contain ``ts_event`` (datetime[ns]),
        ``bid_px_00``/``ask_px_00``, and size columns
        ``bid_sz_00``..``bid_sz_09`` / ``ask_sz_00``..``ask_sz_09``.
        If ``instrument_id`` is present, markouts are computed per
        instrument so searchsorted never crosses boundaries.
    horizons_ns : sequence of int, optional
        Forward horizons in nanoseconds. Defaults span 1 μs to 10 s.
    sample_n : int
        Total ticks to sample across all instruments.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        One row per sampled tick with columns: ``ts_recv``, ``mid_price``,
        ``book_imbalance``, ``obi_level0``, ``instrument_id`` (if present),
        and ``markout_{h}ns`` / ``markout_bps_{h}ns`` for each horizon.
    """
    if horizons_ns is None:
        horizons_ns = _DEFAULT_HORIZONS_NS
    horizons_ns = [int(h) for h in horizons_ns]

    has_instrument = "instrument_id" in df.columns
    groups = (
        df.partition_by("instrument_id", as_dict=True, maintain_order=True)
        if has_instrument
        else {(None,): df}
    )

    total_rows = df.height
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    for key, gdf in groups.items():
        if gdf.height == 0:
            continue

        # Sort inside the group so searchsorted on ts_ns is valid.
        gdf = gdf.sort("ts_event")
        n = gdf.height

        ts_ns = gdf["ts_event"].dt.timestamp("ns").to_numpy()
        bid0 = gdf["bid_px_00"].to_numpy()
        ask0 = gdf["ask_px_00"].to_numpy()
        mid = (bid0 + ask0) / 2.0

        bid_sz0 = gdf["bid_sz_00"].to_numpy().astype(np.float64)
        ask_sz0 = gdf["ask_sz_00"].to_numpy().astype(np.float64)
        denom0 = bid_sz0 + ask_sz0
        with np.errstate(divide="ignore", invalid="ignore"):
            obi_level0 = np.where(denom0 > 0, (bid_sz0 - ask_sz0) / denom0, np.nan)

        # Full-book imbalance aggregates all 10 levels.
        bid_sz_full = np.zeros(n, dtype=np.float64)
        ask_sz_full = np.zeros(n, dtype=np.float64)
        for lvl in _LEVELS:
            b = f"bid_sz_{lvl:02d}"
            a = f"ask_sz_{lvl:02d}"
            if b in gdf.columns:
                bid_sz_full += gdf[b].to_numpy().astype(np.float64)
            if a in gdf.columns:
                ask_sz_full += gdf[a].to_numpy().astype(np.float64)
        denom_full = bid_sz_full + ask_sz_full
        with np.errstate(divide="ignore", invalid="ignore"):
            book_imbalance = np.where(
                denom_full > 0,
                (bid_sz_full - ask_sz_full) / denom_full,
                np.nan,
            )

        group_share = n / total_rows
        group_sample = min(n, max(1, int(round(sample_n * group_share))))
        sample_idx = np.sort(rng.choice(n, size=group_sample, replace=False))

        ts_sampled = pd.to_datetime(ts_ns[sample_idx], utc=True).tz_convert(
            "US/Eastern"
        )

        records: dict = {
            "ts_recv": ts_sampled,
            "mid_price": mid[sample_idx],
            "book_imbalance": book_imbalance[sample_idx],
            "obi_level0": obi_level0[sample_idx],
        }
        if has_instrument:
            inst = key[0] if isinstance(key, tuple) else key
            records["instrument_id"] = np.full(sample_idx.shape, inst)

        current_mid = mid[sample_idx]
        for h in horizons_ns:
            future_ts = ts_ns[sample_idx] + h
            future_idx = np.clip(
                np.searchsorted(ts_ns, future_ts, side="left"), 0, n - 1
            )
            future_mid = mid[future_idx]
            markout = future_mid - current_mid
            with np.errstate(divide="ignore", invalid="ignore"):
                markout_bps = np.where(
                    current_mid > 0, markout / current_mid * 10_000, np.nan
                )
            records[f"markout_{h}ns"] = markout
            records[f"markout_bps_{h}ns"] = markout_bps

        frames.append(pd.DataFrame(records))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_markout_curves(
    markouts_df: pd.DataFrame,
    signal_col: str = "book_imbalance",
    n_bins: int = 5,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot mean markout (bps) vs horizon, binned by a signal.

    Parameters
    ----------
    markouts_df : pd.DataFrame
        Output of :func:`compute_markouts`.
    signal_col : {"book_imbalance", "obi_level0"} or any numeric column
        Column to bin on. Defaults to ``"book_imbalance"``.
    n_bins : int
        Number of quantile bins.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if signal_col not in markouts_df.columns:
        raise ValueError(
            f"signal_col={signal_col!r} not in markouts_df; "
            f"available: {list(markouts_df.columns)}"
        )

    tmp = markouts_df.copy()
    tmp["_bin"] = pd.qcut(tmp[signal_col], n_bins, duplicates="drop")

    bps_cols = [c for c in tmp.columns if c.startswith("markout_bps_") and c.endswith("ns")]
    horizons = sorted(int(c[len("markout_bps_"):-2]) for c in bps_cols)
    ordered_cols = [f"markout_bps_{h}ns" for h in horizons]

    binned = tmp.groupby("_bin", observed=True)[ordered_cols].mean()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    cmap = plt.cm.RdYlGn
    n = max(len(binned) - 1, 1)
    for i, (label, row) in enumerate(binned.iterrows()):
        ax.plot(
            horizons,
            row[ordered_cols].values,
            linewidth=2,
            markersize=4,
            label=str(label),
            color=cmap(i / n),
        )

    # symlog keeps the μs→s log spread but still renders x=0.
    # positive = [h for h in horizons if h > 0]
    # linthresh = positive[0] if positive else 1
    # ax.set_xscale("symlog", linthresh=linthresh)
    # ax.set_xlabel("Time (ns, symlog)")
    ax.set_xscale("log")
    ax.set_xlabel("Time (ns, log)")
    ax.set_ylabel("Markout (bps)")
    ax.set_title(f"Markout Curves by {signal_col}")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.25, linestyle=":")
    ax.legend(title=signal_col, fontsize=8, loc="best")
    fig.tight_layout()
    return fig
