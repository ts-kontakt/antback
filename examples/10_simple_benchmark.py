import time
import numpy as np
from talipp.indicators import SMA
import antback as ab


def timeit(func):
    """Decorator to measure execution time of a function."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"\t'{func.__name__}' executed in: {end - start:.4f}s")
        return result

    return wrapper


@timeit
def sma_cross_np(rows, sym, fast=10, slow=30):
    """SMA crossover using NumPy."""
    port = ab.Portfolio(10_000, single=True, fees=0.0015, allow_fractional=True)
    roll = ab.RollingArray(size=slow)
    cross = ab.new_cross_func()

    for dt, px in rows:
        roll.append(px)
        vals = roll.values()
        sig = None
        if len(vals) >= slow:
            fast_ma = np.mean(vals[-fast:])
            slow_ma = np.mean(vals[-slow:])
            side = cross(fast_ma, slow_ma)

            if side == "up":
                sig = "buy"
            elif side == "down":
                sig = "sell"

        port.process(sig, sym, dt, px)

    rep = port.basic_report(show=False)
    print(rep[1])
    return rep


@timeit
def sma_cross_talipp(rows, sym, fast=10, slow=30):
    """SMA crossover using talipp SMA indicator."""
    port = ab.Portfolio(10_000, single=True, fees=0.0015, allow_fractional=True)
    cross = ab.new_cross_func()
    sma_fast, sma_slow = SMA(period=fast), SMA(period=slow)

    for dt, px in rows:
        sma_fast.add(px)
        sma_slow.add(px)
        sig = None

        if sma_fast[-1] and sma_slow[-1]:
            side = cross(sma_fast[-1], sma_slow[-1])
            if side == "up":
                sig = "buy"
            elif side == "down":
                sig = "sell"

        port.process(sig, sym, dt, px)

    rep = port.basic_report(show=False)
    print(rep[1])
    return rep


@timeit
def sma_cross_antback(rows, sym, fast=10, slow=30):
    """SMA crossover using Antback stateful SMA functions."""
    port = ab.Portfolio(10_000, single=True, fees=0.0015, allow_fractional=True)
    cross = ab.new_cross_func()
    sma_fast = ab.create_sma_func(period=fast)
    sma_slow = ab.create_sma_func(period=slow)

    for dt, px in rows:
        fast_val = sma_fast(px)
        slow_val = sma_slow(px)
        sig = None

        if fast_val and slow_val:
            side = cross(fast_val, slow_val)
            if side == "up":
                sig = "buy"
            elif side == "down":
                sig = "sell"

        port.process(sig, sym, dt, px)

    rep = port.basic_report(show=False)
    print(rep[1])
    return rep


def main():
    import yfinance as yf

    sym = "QQQ"
    closes = yf.Ticker(sym).history(period="30y")["Close"]
    rows = tuple(zip(closes.index, closes.values))

    sma_cross_np(rows, sym)
    sma_cross_talipp(rows, sym)
    sma_cross_antback(rows, sym)


if __name__ == "__main__":
    main()
