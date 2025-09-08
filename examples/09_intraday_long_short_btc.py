#!/usr/bin/python
# coding=utf-8
"""
Backtest a Simple Moving Average (SMA) crossover strategy on 15-minute BTC/USD price data.

Strategy Rules:
- Go long when the fast SMA crosses above the slow SMA.
- Go short when the fast SMA crosses below the slow SMA.
- Close an existing position if a new signal is opposite.
"""

import pandas as pd

import antback as ab

START_CAPITAL = 10_000.0
MARGIN_REQ = 0.1
FAST_SMA = 20
SLOW_SMA = 40
TICKER = "BTC-USD"
DATA_PATH = "btc_1m.csv"  # Path to historical 1-min price data


def backtest_sma_strategy(price_series, symbol, fast_period, slow_period):
    # Initialize trading account
    account = ab.CFDAccount(cash=START_CAPITAL,
                            margin_requirement=MARGIN_REQ,
                            leverage=3,
                            warn=False)

    # SMA calculators and crossover detector
    detect_crossover = ab.new_cross_func()
    fast_sma = ab.create_sma_func(period=fast_period)
    slow_sma = ab.create_sma_func(period=slow_period)

    # Iterate over data points
    for timestamp, price in price_series:
        
        # Calculate current SMA values (returns None until sufficient data)
        fast_value = fast_sma(price)
        slow_value = slow_sma(price)
    
        action = "update"
        # Generate signals when SMAs are available
        if fast_value and slow_value:
            cross = detect_crossover(fast_value, slow_value)
            if cross == "up":
                action = "long"
            elif cross == "down":
                action = "short"

        # Handle position reversals
        if account.has_position():
            current_pos = account.position[-1]
            if action in ("long", "short") and action != current_pos:
                account.process("close", symbol, timestamp, price, log_msg="Reversal")

        account.process(action, symbol, timestamp, price)
    # close remaining position
    account.process('close', symbol, timestamp, price)
    return account


def run_backtest():
    """Load data, run backtest, and print performance reports."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['close_time'] = pd.to_datetime(df['close_time'])
    df.set_index('close_time', inplace=True)

    # Resample to 15-minute intervals
    resampled = df['close'].resample('15min').ohlc()
    price_data = tuple(zip(resampled.index, resampled.close))

    
    account = backtest_sma_strategy(price_data, TICKER, FAST_SMA, SLOW_SMA)

    print("\n--- Backtest Summary ---")
    print(account)
    account.basic_report()
    account.full_report(title=f"\n{TICKER} - SMA Crossover Backtest ({FAST_SMA}/{SLOW_SMA})")  # Outputs report.html


if __name__ == "__main__":
    run_backtest()
