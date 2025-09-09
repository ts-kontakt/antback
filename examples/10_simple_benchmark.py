import time

import numpy as np
from talipp.indicators import SMA

import antback as ab


def timer_decorator(func):

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"\t'{func.__name__}' executed in: {end_time - start_time:.4f}s")
        return result

    return wrapper


@timer_decorator
def sma_cross_np(data, symbol, fast=10, slow=30):
    # Initialize portfolio and data storage
    port = ab.Portfolio(10_000, single=True, fees=0.0015, allow_fractional=True)
    prices = ab.RollingArray(size=slow)
    cross = ab.new_cross_func()

    for date, price in data:
        prices.append(price)
        price_history = prices.values()
        signal = None  # Reset signal for each iteration
        if len(price_history) >= slow:
            fast_ma = np.mean(price_history[-fast:])
            slow_ma = np.mean(price_history[-slow:])
            direction = cross(fast_ma, slow_ma)

            if direction == "up":
                signal = 'buy'
            elif direction == "down":
                signal = 'sell'
        port.process(signal, symbol, date, price)

    # Generate reports
    rep = port.basic_report(show=False)
    print(rep[1])
    return rep


@timer_decorator
def sma_cross_talipp(data, symbol, fast=10, slow=30):
    # Initialize portfolio and indicators
    port = ab.Portfolio(10_000, single=True, fees=0.0015, allow_fractional=True)
    cross = ab.new_cross_func()
    fast_sma, slow_sma = SMA(period=fast), SMA(period=slow)

    for date, price in data:  # For dataframe use .itertuples()
        # Update SMA indicators with latest price
        fast_sma.add(price)
        slow_sma.add(price)
        signal = None

        # Check if we have enough data for valid SMA values
        if fast_sma[-1] and slow_sma[-1]:  # Returns None if not enough data
            direction = cross(fast_sma[-1], slow_sma[-1])
            if direction == "up":
                signal = "buy"
            elif direction == "down":
                signal = "sell"

        port.process(signal, symbol, date, price)
    rep = port.basic_report(show=False)
    print(rep[1])
    return rep


@timer_decorator
def sma_cross_antback(data, symbol, fast=10, slow=30):
    # Initialize portfolio and indicators
    port = ab.Portfolio(10_000, single=True, fees=0.0015, allow_fractional=True)
    cross = ab.new_cross_func()
    #Antback stateful clousure based functions that does not need data separate rolling structures
    sma_fast = ab.create_sma_func(period=fast)
    sma_slow = ab.create_sma_func(period=slow)

    for date, price in data:  # For dataframe use .itertuples()
        # Update SMA indicators with latest price
        fast_val = sma_fast(price)
        slow_val = sma_slow(price)
        signal = None

        # Check if we have enough data for valid SMA values
        if fast_val and slow_val:  # Returns None if not enough data
            direction = cross(fast_val, slow_val)
            if direction == "up":
                signal = "buy"
            elif direction == "down":
                signal = "sell"

        port.process(signal, symbol, date, price)
    rep = port.basic_report(show=False)
    print(rep[1])
    return rep


def main():
    import yfinance as yf
    symbol = "QQQ"
    data = yf.Ticker(symbol).history(period="30y")["Close"]

    data_rows = tuple(zip(data.index, data.values))

    sma_cross_np(data_rows, symbol)
    sma_cross_talipp(data_rows, symbol)
    sma_cross_antback(data_rows, symbol)


if __name__ == "__main__":
    main()
