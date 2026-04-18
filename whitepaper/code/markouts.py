import sys, os
from data_collection.data_adapters import DatabentoGenericAdapter
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from saf_lib.execution import compute_markouts, plot_markout_curves
import polars as pl
import numpy as np
import asyncio
load_dotenv()

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
    LATENCY = 50_000 #50us in ns
    REPLENISHMENT = 30.0 #poisson rate of replenishment using wall-clock
    PARENT_QTY = 5_000
    SEED = 42

    # Get Data adn select to relavent columns
    df, map = asyncio.run(get_data())
    print(f"File size (gb): {df.estimated_size('gb'):.2f}")
    cols = ["ts_event", "instrument_id"] + [
        col for n in range(0, 10)
        for col in [f"bid_px_{str(n).zfill(2)}", f"ask_px_{str(n).zfill(2)}", f"bid_sz_{str(n).zfill(2)}", f"ask_sz_{str(n).zfill(2)}"]
    ]
    df = df.select(cols)

    # Compute markouts at multiple horizons (sampled for efficiency)
    horizon_ns = np.arange(0, 5_000_000, 10_000) # 0 to 5000us (5ms) in steps of 10us
    markouts = compute_markouts(df, horizons_ns=horizon_ns, sample_n=100_000, seed=SEED)
    print(f"Computed markouts for {len(markouts):,} sampled ticks")

    # Markout curves conditioned on full-book imbalance
    plot_markout_curves(markouts, signal_col='book_imbalance', n_bins=4)
    plt.savefig(f'plots/full_book_markout_{SEED}', dpi=300, bbox_inches='tight')
    plt.show()

    # Markout curves conditioned on top level imbalance only
    plot_markout_curves(markouts, signal_col='obi_level0', n_bins=4)
    plt.savefig(f'plots/top_level_markout_{SEED}', dpi=300, bbox_inches='tight')
    plt.show()