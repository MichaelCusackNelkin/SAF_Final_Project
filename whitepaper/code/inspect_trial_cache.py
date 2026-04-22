"""Inspect a trial cache parquet and plot OBI vs constant scatter plots."""
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
import polars as pl

from trials import _trial_cache_key


def main() -> None:
    """Load OBI and constant parquets for a cached trial and scatter-plot them.

    The input parquet's sidecar JSON is used to recover the simulation
    parameters, and the partner (hybrid <-> constant) parquet is located by
    recomputing the MD5 cache key with the opposite ``method``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Path to a trial cache parquet.")
    args = parser.parse_args()

    in_path: Path = args.parquet
    cache_dir = in_path.parent
    params_path = cache_dir / f"{in_path.stem}.json"
    with open(params_path) as f:
        params = json.load(f)

    print("Parameters:")
    for k in sorted(params):
        print(f"  {k}: {params[k]}")

    hybrid_key = _trial_cache_key(**{**params, "method": "hybrid"})
    const_key = _trial_cache_key(**{**params, "method": "constant"})

    hybrid_path = cache_dir / f"{hybrid_key}.parquet"
    const_path = cache_dir / f"{const_key}.parquet"

    hybrid = pl.read_parquet(hybrid_path)
    const = pl.read_parquet(const_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    axes[0].scatter(
        hybrid["parent_filled"], hybrid["parent_arrival_cost"],
        color="tab:orange", alpha=0.4, s=8,
    )
    axes[0].set_title("Hybrid")
    axes[0].set_xlabel("parent_filled")
    axes[0].set_ylabel("parent_arrival_cost (bps)")

    axes[1].scatter(
        const["parent_filled"], const["parent_arrival_cost"],
        color="tab:blue", alpha=0.4, s=8,
    )
    axes[1].set_title("Constant")
    axes[1].set_xlabel("parent_filled")

    fig.suptitle(
        f"{params.get('ticker', '?')}: parent_filled vs parent_arrival_cost"
    )
    fig.tight_layout()

    out_path = cache_dir / f"scatter_{in_path.stem}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Hybrid parquet:   {hybrid_path}")
    print(f"Constant parquet: {const_path}")


if __name__ == "__main__":
    main()
