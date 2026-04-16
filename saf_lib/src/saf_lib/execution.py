"""
execution.py — Mid-price markout analysis for MBP-10 tick data.

Exposes two functions:
    - compute_markouts: sample ticks and measure forward mid-price moves
      at configurable horizons (nanoseconds, so μs → s is supported).
    - plot_markout_curves: plot mean markout (bps) vs horizon, binned by
      a signal column (``book_imbalance`` or ``obi_level0``).

All heavy calculations run in polars; matplotlib is used only to render
the final pre-aggregated curves.
"""

from __future__ import annotations

import numpy as np
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
) -> pl.DataFrame:
    """Compute forward mid-price markouts at multiple horizons.

    For each sampled tick the markout at horizon *h* (ns) is the
    mid-price change from the current tick to the first tick whose
    timestamp is ≥ current_ts + *h*. If no such future tick exists
    within the instrument's data, the last observed mid is used.

    Parameters
    ----------
    df : pl.DataFrame
        MBP-10 polars frame. Must contain ``ts_event`` (datetime[ns]),
        ``bid_px_00``/``ask_px_00``, and size columns
        ``bid_sz_00``..``bid_sz_09`` / ``ask_sz_00``..``ask_sz_09``.
        If ``instrument_id`` is present, markouts are computed per
        instrument so the forward-join never crosses boundaries.
    horizons_ns : sequence of int, optional
        Forward horizons in nanoseconds. Defaults span 1 μs to 10 s.
    sample_n : int
        Total ticks to sample across all instruments.
    seed : int
        RNG seed.

    Returns
    -------
    pl.DataFrame
        One row per sampled tick with columns: ``ts_recv``, ``mid_price``,
        ``book_imbalance``, ``obi_level0``, ``instrument_id`` (if present),
        and ``markout_{h}ns`` / ``markout_bps_{h}ns`` for each horizon.
    """
    if horizons_ns is None:
        horizons_ns = _DEFAULT_HORIZONS_NS
    horizons_ns = [int(h) for h in horizons_ns]

    has_instrument = "instrument_id" in df.columns

    bid_sz_cols = [f"bid_sz_{lvl:02d}" for lvl in _LEVELS if f"bid_sz_{lvl:02d}" in df.columns]
    ask_sz_cols = [f"ask_sz_{lvl:02d}" for lvl in _LEVELS if f"ask_sz_{lvl:02d}" in df.columns]

    bid_full = pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in bid_sz_cols])
    ask_full = pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in ask_sz_cols])
    denom_full = bid_full + ask_full
    bid0 = pl.col("bid_sz_00").cast(pl.Float64)
    ask0 = pl.col("ask_sz_00").cast(pl.Float64)
    denom0 = bid0 + ask0

    enriched = df.with_columns(
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2.0).alias("mid_price"),
        pl.when(denom_full > 0)
          .then((bid_full - ask_full) / denom_full)
          .otherwise(None)
          .alias("book_imbalance"),
        pl.when(denom0 > 0)
          .then((bid0 - ask0) / denom0)
          .otherwise(None)
          .alias("obi_level0"),
    )

    sort_keys = ["instrument_id", "ts_event"] if has_instrument else ["ts_event"]
    enriched = enriched.sort(sort_keys)

    rng = np.random.default_rng(seed)
    if has_instrument:
        total_rows = enriched.height
        if total_rows == 0:
            return pl.DataFrame()
        parts = enriched.partition_by("instrument_id", as_dict=True, maintain_order=True)
        sampled_parts: list[pl.DataFrame] = []
        for _, gdf in parts.items():
            n = gdf.height
            if n == 0:
                continue
            share = n / total_rows
            k = min(n, max(1, int(round(sample_n * share))))
            group_seed = int(rng.integers(0, 2**31 - 1))
            sampled_parts.append(gdf.sample(n=k, seed=group_seed))
        if not sampled_parts:
            return pl.DataFrame()
        sampled = pl.concat(sampled_parts)
    else:
        n = enriched.height
        if n == 0:
            return pl.DataFrame()
        k = min(n, sample_n)
        sampled = enriched.sample(n=k, seed=seed)

    base_cols = ["ts_event", "mid_price", "book_imbalance", "obi_level0"]
    if has_instrument:
        base_cols = ["instrument_id"] + base_cols
    sampled = sampled.select(base_cols).sort(sort_keys).with_row_index("_rid")

    lookup_cols = (["instrument_id"] if has_instrument else []) + ["ts_event", "mid_price"]
    lookup = enriched.select(lookup_cols).rename({"mid_price": "_future_mid"})

    # Fallback when target_ts exceeds the last observed ts in a group:
    # join_asof(forward) returns null there, so we coalesce with the
    # instrument's final mid to match the original clip-to-last behavior.
    if has_instrument:
        last_mid = enriched.group_by("instrument_id").agg(
            pl.col("mid_price").last().alias("_last_mid")
        )
    else:
        last_mid_scalar = enriched["mid_price"].last()

    for h in horizons_ns:
        target = sampled.with_columns(
            (pl.col("ts_event") + pl.duration(nanoseconds=h)).alias("_target_ts")
        )
        target_sort = ["instrument_id", "_target_ts"] if has_instrument else ["_target_ts"]
        target_sorted = target.sort(target_sort)

        join_args = dict(
            left_on="_target_ts",
            right_on="ts_event",
            strategy="forward",
        )
        if has_instrument:
            join_args["by"] = "instrument_id"

        joined = target_sorted.join_asof(lookup, **join_args)

        if has_instrument:
            joined = (
                joined.join(last_mid, on="instrument_id", how="left")
                .with_columns(pl.col("_future_mid").fill_null(pl.col("_last_mid")))
                .drop("_last_mid")
            )
        else:
            joined = joined.with_columns(
                pl.col("_future_mid").fill_null(pl.lit(last_mid_scalar))
            )

        joined = joined.sort("_rid").select("_rid", "_future_mid")

        sampled = (
            sampled.join(joined, on="_rid", how="left")
            .with_columns((pl.col("_future_mid") - pl.col("mid_price")).alias(f"markout_{h}ns"))
            .with_columns(
                pl.when(pl.col("mid_price") > 0)
                  .then(pl.col(f"markout_{h}ns") / pl.col("mid_price") * 10_000)
                  .otherwise(None)
                  .alias(f"markout_bps_{h}ns")
            )
            .drop("_future_mid")
        )

    ts_tz = sampled.schema["ts_event"].time_zone
    if ts_tz is None:
        ts_expr = pl.col("ts_event").dt.replace_time_zone("UTC").dt.convert_time_zone("US/Eastern")
    else:
        ts_expr = pl.col("ts_event").dt.convert_time_zone("US/Eastern")

    result = sampled.with_columns(ts_expr.alias("ts_recv")).drop("_rid", "ts_event")

    front = ["ts_recv", "mid_price", "book_imbalance", "obi_level0"]
    if has_instrument:
        front.append("instrument_id")
    horizon_cols: list[str] = []
    for h in horizons_ns:
        horizon_cols.extend([f"markout_{h}ns", f"markout_bps_{h}ns"])
    return result.select(front + horizon_cols)


def plot_markout_curves(
    markouts_df: pl.DataFrame,
    signal_col: str = "book_imbalance",
    n_bins: int = 5,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot mean markout (bps) vs horizon, binned by a signal.

    Parameters
    ----------
    markouts_df : pl.DataFrame
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
            f"available: {markouts_df.columns}"
        )

    bps_cols = [c for c in markouts_df.columns if c.startswith("markout_bps_") and c.endswith("ns")]
    horizons = sorted(int(c[len("markout_bps_"):-2]) for c in bps_cols)
    ordered_cols = [f"markout_bps_{h}ns" for h in horizons]

    # Quantile breakpoints as absolute floats keeps bin labels numerically
    # sortable, and avoids relying on categorical ordering from qcut.
    qs = [i / n_bins for i in range(1, n_bins)]
    signal = markouts_df.get_column(signal_col).drop_nulls().drop_nans()
    breaks = sorted({float(signal.quantile(q)) for q in qs})

    binned = (
        markouts_df
        .with_columns(pl.col(signal_col).cut(breaks, include_breaks=False).alias("_bin"))
        .drop_nulls("_bin")
        .group_by("_bin", maintain_order=False)
        .agg(
            [pl.col(c).mean() for c in ordered_cols]
            + [pl.col(signal_col).min().alias("_lo")]
        )
        .sort("_lo")
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    cmap = plt.cm.RdYlGn
    n_rows = binned.height
    denom = max(n_rows - 1, 1)
    labels = binned.get_column("_bin").cast(pl.String).to_list()
    values = binned.select(ordered_cols).to_numpy()
    for i, label in enumerate(labels):
        ax.plot(
            horizons,
            values[i],
            linewidth=2,
            markersize=4,
            label=label,
            color=cmap(i / denom),
        )

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
