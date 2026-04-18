from data_collection.data_adapters import DatabentoGenericAdapter
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import hashlib
import json
import polars as pl
import os
load_dotenv()


def _trial_cache_key(
    ticker: str,
    latency: float,
    replenishment: float,
    parent_qty: int,
    depth_lmt: str,
    threshold: float,
    horizon: float,
    min_ns: float,
    min_shares: int,
    max_size: int,
    wait_ns: float,
    n_chunks: int,
    method: str,
    n_trials: int,
) -> str:
    """Compute an MD5 hash over the trial parameters for cache keying.

    Parameters
    ----------
    ticker : str
        Ticker symbol used for the trial.
    latency : float
        Simulated latency in nanoseconds.
    replenishment : float
        Poisson rate of book replenishment (wall-clock).
    parent_qty : int
        Total parent order quantity.
    depth_lmt : str
        Zero-padded depth level string (e.g. "03").
    threshold : float
        OBI entry threshold (absolute value).
    horizon : float
        Trading horizon in nanoseconds.
    min_ns : float
        Minimum wait between fills for OBI filler, in nanoseconds.
    min_shares : int
        Minimum shares traded between fills for OBI filler.
    max_size : int
        Maximum child order size.
    wait_ns : float
        Constant filler inter-child wait, in nanoseconds.
    n_chunks : int
        Number of chunks used by the constant filler.
    method : str
        Filler method name (e.g. "OBI" or "constant").
    n_trials : int
        Number of trials in the run.

    Returns
    -------
    str
        Hex digest of the MD5 hash of the serialized parameter dict.
    """
    payload = {
        "ticker": ticker,
        "latency": latency,
        "replenishment": replenishment,
        "parent_qty": parent_qty,
        "depth_lmt": depth_lmt,
        "threshold": threshold,
        "horizon": horizon,
        "min_ns": min_ns,
        "min_shares": min_shares,
        "max_size": max_size,
        "wait_ns": wait_ns,
        "n_chunks": n_chunks,
        "method": method,
        "n_trials": n_trials,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def load_or_run(
    method: str,
    trial_args: list,
    n_trials: int,
    df: pl.DataFrame,
    cache_dir: Path,
    max_workers: int,
    **cache_kwargs,
) -> pl.DataFrame:
    """Load cached trial results or run the ProcessPoolExecutor trial loop.

    Parameters
    ----------
    method : str
        Filler method name (e.g. "OBI" or "constant"). Used for the cache
        key, the tqdm progress-bar description, and caching.
    trial_args : list
        List of argument tuples passed to ``run_trial`` (one per trial).
    n_trials : int
        Number of trials to run; also used as the ``tqdm`` total.
    df : pl.DataFrame
        Market data DataFrame broadcast to worker processes via
        ``init_worker``.
    cache_dir : pathlib.Path
        Directory where cached parquet results are stored.
    max_workers : int
        Number of worker processes for the ``ProcessPoolExecutor``.
    **cache_kwargs
        Scalar trial parameters forwarded to :func:`_trial_cache_key`
        (all keyword arguments except ``method`` and ``n_trials``).

    Returns
    -------
    pl.DataFrame
        DataFrame of trial results with a ``lifetime_ns`` column
        appended.
    """
    from concurrent.futures import ProcessPoolExecutor
    from saf_lib.simulation import init_worker, run_trial

    key = _trial_cache_key(method=method, n_trials=n_trials, **cache_kwargs)
    path = cache_dir / f"{key}.parquet"

    if path.exists():
        print(f"[cache hit] {method}: loading {path}")
        return pl.read_parquet(path)

    print(f"[cache miss] {method}: running {n_trials} trials -> {path}")
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=init_worker, initargs=(df,)
    ) as ex:
        results = [
            c
            for c in tqdm(
                ex.map(run_trial, trial_args), total=n_trials, desc=method
            )
            if c is not None
        ]

    out = pl.DataFrame(results)
    out = out.with_columns(
        pl.col("child_fill_times").list.sum().alias("lifetime_ns")
    )
    out.write_parquet(path)
    return out

async def get_data() -> pl.DataFrame:
    adapter = DatabentoGenericAdapter(os.getenv("DATABENTO_API_KEY"))
    df, map = await adapter.get_data(
        tickers=["CSCO", "AMZN", "DRH"],
        start_date="2026-03-25",
        end_date="2026-03-31",
        dataset="XNAS.ITCH",
        schema="MBP-10",
        cache_only=True
    )
    map = {v: k for k,v in map.items()}
    return df, map



if __name__ == "__main__":
    import asyncio
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    from saf_lib.simulation import *

    # -- General Params --
    TICKER = "AMZN"
    LATENCY = 100 * 1e3                 #100us, ns
    REPLENISHMENT = 10.0                #poisson rate of replenishment using wall-clock
    PARENT_QTY = 5_000
    DEPTH_LMT = "03"
    N_TRIALS = 10
    MAX_WORKERS = 5

    # -- OBI Params --
    MIN_NS = 100 * 1e6                  #100ms, waiting time between fills for OBI filler, ns
    MIN_SHARES = 0                      # waiting time in shares traded between fills for OBI filler
    THRESHOLD = 0.90                    # OBI threshold for entry, +- for buys and sells
    HORIZON = 120 * 1e9                  # HALF-HORIZON 30s, allowed trading horizon, ns
    MAX_SIZE = 500                      # Maximum child order sizes

    # -- Constant Params --
    WAIT_NS = 1e9
    N_CHUNKS = 30

    # -- Cache Dir --
    CACHE_DIR = Path("trial_cache")
    CACHE_DIR.mkdir(exist_ok=True)

    # Get Data and select to relavent columns
    df, map = asyncio.run(get_data())
    cols = ["ts_event", "instrument_id", "action", "price", "size"] + [
        col for n in range(0, 10)
        for col in [f"bid_px_{str(n).zfill(2)}", f"ask_px_{str(n).zfill(2)}", f"bid_sz_{str(n).zfill(2)}", f"ask_sz_{str(n).zfill(2)}"]
    ]
    df = df.select(cols)
    df = df.filter(pl.col("instrument_id") == map[TICKER]).with_columns(
        pl.col("ts_event").dt.timestamp("ns").alias("ts_nano")
    )
    df = df.with_columns(((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2).alias("mid"))
    print(f"Sorted: {df.sort("ts_nano").equals(df)}")
    print(f"Estimated Data Size: {df.estimated_size('gb'):.2f} GB")

    """Parallelized Trials"""
    shared_kwargs = dict(
        ticker=TICKER,
        latency=LATENCY,
        replenishment=REPLENISHMENT,
        parent_qty=PARENT_QTY,
        depth_lmt=DEPTH_LMT,
        threshold=THRESHOLD,
        horizon=HORIZON,
        min_ns=MIN_NS,
        min_shares=MIN_SHARES,
        max_size=MAX_SIZE,
        wait_ns=WAIT_NS,
        n_chunks=N_CHUNKS,
    )

    # ===========
    # OBI Trial
    # ===========
    METHOD = "OBI"
    trial_args = [(LATENCY, REPLENISHMENT, PARENT_QTY, DEPTH_LMT,
                THRESHOLD, HORIZON, MIN_NS, MIN_SHARES, MAX_SIZE, WAIT_NS, N_CHUNKS, METHOD)] * N_TRIALS
    cost_obi = load_or_run(
        method=METHOD,
        trial_args=trial_args,
        n_trials=N_TRIALS,
        df=df,
        cache_dir=CACHE_DIR,
        max_workers=MAX_WORKERS,
        **shared_kwargs,
    )

    #================
    # Constant Trial
    #================
    METHOD = "constant"
    trial_args = [(LATENCY, REPLENISHMENT, PARENT_QTY, DEPTH_LMT,
                THRESHOLD, HORIZON, MIN_NS, MIN_SHARES, MAX_SIZE, WAIT_NS, N_CHUNKS, METHOD)] * N_TRIALS
    cost_constant = load_or_run(
        method=METHOD,
        trial_args=trial_args,
        n_trials=N_TRIALS,
        df=df,
        cache_dir=CACHE_DIR,
        max_workers=MAX_WORKERS,
        **shared_kwargs,
    )

    # -- Descriptive Stats --
    print("\nOBI Fill Costs")
    print(cost_obi.describe())
    print("\nConstant Fill Costs")
    print(cost_constant.describe())

    #===========
    # Plotting
    #===========

    plot_obi = cost_obi
    plot_constant = cost_constant

    # -- Parent Costs --
    log = True
    plt.hist(plot_obi["parent_arrival_cost"], bins=25, edgecolor="black", color="tab:orange", alpha=0.7, label="OBI", log=log)
    plt.hist(plot_constant["parent_arrival_cost"], bins=25, edgecolor="black", color="tab:blue", alpha=0.7, label="Constant", log=log)
    plt.xlabel("Cost per Share (bps)")
    plt.ylabel("Count")
    plt.title("Parent Cost Per Share Comparison: OBI vs Constant IOC")
    plt.legend()
    plt.savefig(f"plots/parent_cost_{N_TRIALS}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # -- Child Costs --
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    # -- Means --
    ax[0].hist(plot_obi["mean_child_arrival_cost"], bins=25, edgecolor="black", color="tab:orange", alpha=0.7, label="OBI")
    ax[0].hist(plot_constant["mean_child_arrival_cost"], bins=25, edgecolor="black", color="tab:blue", alpha=0.7, label="Constant")
    ax[0].set_xlabel("Cost per Share (bps)")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Mean Arrival Cost")
    ax[0].legend()

    # -- Medians --
    ax[1].hist(plot_obi["median_child_arrival_cost"], bins=25, edgecolor="black", color="tab:orange", alpha=0.7, label="OBI")
    ax[1].hist(plot_constant["median_child_arrival_cost"], bins=25, edgecolor="black", color="tab:blue", alpha=0.7, label="Constant")
    ax[1].set_xlabel("Cost per Share (bps)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Median Arrival Cost")
    ax[1].legend()

    plt.suptitle("Child Arrival Cost Per Share Comparison: OBI vs Constant IOC ")
    plt.tight_layout()
    plt.savefig(f"plots/child_cost_{N_TRIALS}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # -- Fills --
    from matplotlib.ticker import PercentFormatter
    plt.hist(
        plot_obi["parent_filled"],
        weights=np.ones(len(plot_obi["parent_filled"])) / len(plot_obi["parent_filled"]),
        bins=25, edgecolor="black", color="tab:orange", alpha=0.7, label="OBI"
        )
    plt.hist(
        plot_constant["parent_filled"],
        weights=np.ones(len(plot_constant["parent_filled"])) / len(plot_constant["parent_filled"]),
        bins=25, edgecolor="black", color="tab:blue", alpha=0.7, label="Constant"
        )
    plt.xlabel("Cost per Share (bps)")
    plt.ylabel("Percent (%)")
    plt.title("Fill Reliability Comparison: OBI vs Constant IOC")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.legend()
    plt.savefig(f"plots/fill_reliability_{N_TRIALS}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # -- Parent Lifetimes --
    plt.hist(
        plot_obi["lifetime_ns"] / 1e9,
        weights=np.ones(len(plot_obi["lifetime_ns"])) / len(plot_obi["lifetime_ns"]),
        bins=25, edgecolor="black", color="tab:orange", alpha=0.7, label="OBI"
        )
    plt.hist(
        plot_constant["lifetime_ns"] / 1e9,
        weights=np.ones(len(plot_constant["lifetime_ns"])) / len(plot_constant["lifetime_ns"]),
        bins=25, edgecolor="black", color="tab:blue", alpha=0.7, label="Constant"
        )
    plt.xlabel("Lifetime (s)")
    plt.ylabel("Percent (%)")
    plt.title("Parent Order Lifetime Comparison: OBI vs Constant IOC")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.legend()
    plt.savefig(f"plots/parents_lifetimes_{N_TRIALS}.png", dpi=300, bbox_inches='tight')
    plt.show()

