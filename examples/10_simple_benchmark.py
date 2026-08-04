import gc
import time
import numpy as np
from talipp.indicators import SMA
import antback2 as ab


def bench(func, *args, repeats=15, warmup=3, **kwargs):
    """
    Repeated-trial timing: discard `warmup` cold-start runs, disable GC during
    each timed call so a collection pause doesn't land on one run at random,
    then report the median (robust to the occasional scheduler/GC outlier)
    alongside min/max so you can see how noisy a single sample would've been.
    A single untimed call (the original @timeit approach) is not trustworthy
    at these durations (10-100ms) -- see the note at the bottom of this file.
    """
    times = []
    result = None
    for i in range(repeats + warmup):
        # gc.collect()
        # gc.disable()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        # gc.enable()
        if i >= warmup:
            times.append(end - start)
    times.sort()
    median = times[len(times) // 2]
    print(f"\t'{func.__name__}' median={median*1000:7.3f}ms  "
          f"min={times[0]*1000:7.3f}ms  max={times[-1]*1000:7.3f}ms  "
          f"(n={len(times)} trials, single-sample spread would've been "
          f"{(times[-1]-times[0])/median*100:.0f}%)")
    return result


def sma_cross_standard(rows, sym, fast=10, slow=30):
    """SMA crossover using python builtins"""
    port = ab.Portfolio(10_000, single=True, fees=0.0015, allow_fractional=True)
    roll = ab.RollingList(maxlen=slow)
    cross = ab.new_cross_func()

    for dt, px in rows:
        roll.append(px)
        vals = roll.values()
        sig = None
        if len(vals) >= slow:
            fast_ma = sum(vals[-fast:]) / fast
            slow_ma = sum(vals) / slow
            side = cross(fast_ma, slow_ma)

            if side == "up":
                sig = "buy"
            elif side == "down":
                sig = "sell"

        port.process(sig, sym, dt, px)

    return port.basic_report(show=False)


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
            fast_ma = vals[-fast:].mean()
            slow_ma = vals[-slow:].mean()
            side = cross(fast_ma, slow_ma)

            if side == "up":
                sig = "buy"
            elif side == "down":
                sig = "sell"

        port.process(sig, sym, dt, px)

    return port.basic_report(show=False)


def sma_cross_talipp(rows, sym, fast=10, slow=30):
    """SMA crossover using talipp SMA indicator."""
    port = ab.Portfolio(10_000, single=True, fees=0.0015, allow_fractional=True)
    cross = ab.new_cross_func()
    sma_fast, sma_slow = SMA(period=fast), SMA(period=slow)
    for dt, px in rows:
        sma_fast.add(px)
        sma_slow.add(px)
        sig = None

        # bbf.add(px)
        # print(bbf[-1])

        if sma_fast[-1] and sma_slow[-1]:
            side = cross(sma_fast[-1], sma_slow[-1])
            if side == "up":
                sig = "buy"
            elif side == "down":
                sig = "sell"

        port.process(sig, sym, dt, px)

    return port.basic_report(show=False)


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

    return port.basic_report(show=False)


def main():
    import yfinance as yf

    sym = "GOOG"
    closes = yf.Ticker(sym).history(period="20y")["Close"]
    closes = closes[closes > 0]
    rows = tuple(zip(closes.index, closes.values))

    rep = bench(sma_cross_standard, rows, sym)
    print(rep[1])
    rep = bench(sma_cross_np, rows, sym)
    print(rep[1])
    rep = bench(sma_cross_talipp, rows, sym)
    print(rep[1])
    rep = bench(sma_cross_antback, rows, sym)
    print(rep[1])


if __name__ == "__main__":
    main()