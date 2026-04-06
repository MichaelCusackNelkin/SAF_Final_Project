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

if __name__ == "__main__":
    import polars as pl
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    df = pl.DataFrame({
        "level": [1,2,3,4,5,6,7,8,9,10],
        "bid_size": [150,300,200,400,500,600,250,800,1000,1200],
        "bid_px": [149.99, 149.98, 149.95, 149.90, 149.85, 149.80, 149.78, 149.75, 149.70, 149.65],
        "ask_px": [150.02, 150.05, 150.06, 150.09, 150.15, 150.18, 150.20, 150.22, 150.25, 150.30],
        "ask_size": [100, 250, 400, 200, 500, 450, 700, 300, 900, 1500],
    })

    # -- Print the book --
    print("(i)")
    book = L2Book(df)
    print(book)

    # -- 200 share mkt buy --
    print("\n"+"(ii)")
    book = L2Book(df)
    size = 200
    vwap = book.accept_mkt_order("BUY", size)
    print(book)

    # -- Trading Costs Analysis --
    print("\n" + "(iii)")
    book = L2Book(df)
    sizes = [200, 400, 800, 1000, 1500, 2000]
    side = "BUY"
    tcosts = []
    best_px = book.asks[0][0] if side == "BUY" else book.bids[0][0]
    for q in sizes:
        book = L2Book(df)
        vwap = book.accept_mkt_order(side, q)
        arrival_cost = vwap - best_px
        tcosts.append(arrival_cost)
    tcosts = pl.DataFrame(data={
        "order_size": sizes,
        "tcosts": tcosts
        })
    
    # Analyze
    print(tcosts.describe())
    X = sm.add_constant((tcosts["order_size"]).to_numpy())
    y = (tcosts["tcosts"] * 10_000).to_numpy()
    result = sm.OLS(y, X).fit()
    print(result.summary())
    """Significant results because pval low, CI does not contain 0, non-zero slope"""
    
    # Plotting
    plt.scatter(tcosts["order_size"], tcosts["tcosts"] * 10_000, label="True Data")
    x = np.arange(0,2000,1)
    plt.plot(x,
             result.params[1] * x + result.params[0],
             linewidth=1, color='red', linestyle = "--", label=f"Fitted Line: {result.params[1]:.2f} * x + {result.params[0]:.2f}")
    plt.title("Arrival Cost as a Function of Order Size")
    plt.xlabel("Order Size")
    plt.ylabel("Arrival Cost (bps)")
    plt.legend()
    plt.show()