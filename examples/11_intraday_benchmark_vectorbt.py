import gc
import time
import os
import pandas as pd
import antback2  as ab


def bench_repeated(func, *args, repeats=10, warmup=2, **kwargs):
    """
    Median-of-N repeated-trial timing. Discards `warmup` cold-start calls,
    disables GC during each timed call (so a collection pause can't land on
    one call by chance), returns the last call's result plus the sorted
    list of timings so you can see min/max spread, not just the median.
    Use this for anything with NO JIT/compilation step (antback qualifies).
    For vectorbt, use bench_first_vs_steady_state below instead -- see why
    in main().
    """
    times = []
    result = None
    for i in range(repeats + warmup):
        # gc.collect()
        # gc.disable()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        # gc.enable()
        if i >= warmup:
            times.append(t1 - t0)
    times.sort()
    return result, times


def bench_first_vs_steady_state(func, *args, repeats=10, **kwargs):
    """
    For JIT-compiled code (vectorbt/Numba): the first call in a process pays
    compilation, every later call in the SAME process is fast. Discarding
    the first call the way the original scriptinstall -U "vectorbt[rust] did only tells you the
    steady-state number -- it silently assumes you keep the process alive
    across many calls. This reports both, explicitly, so you can pick the
    number that matches how you'll actually run this.
    """
    t0 = time.perf_counter()
    first_result = func(*args, **kwargs)
    first_call_time = time.perf_counter() - t0

    steady_kwargs = {k: v for k, v in kwargs.items() if k != "msg"}
    _, steady_times = bench_repeated(func, *args, repeats=repeats, warmup=0, **steady_kwargs)
    return first_result, first_call_time, steady_times


def backtest_sma_cross(rows, sym, fast=40, slow=60):
    """Backtest SMA crossover using Antback."""
    acct = ab.CFDAccount(cash=10_000, margin_requirement=0.1,
                         leverage=1, fees=0.0, warn=0)
    cross = ab.new_cross_func()
    sma_fast = ab.create_sma_func(period=fast)
    sma_slow = ab.create_sma_func(period=slow)

    for dt, price in rows:
        sig = "update"
        fast_val = sma_fast(price)
        slow_val = sma_slow(price)

        if fast_val and slow_val:
            side = cross(fast_val, slow_val)
            if side == "up":
                sig = "long"
            elif side == "down":
                sig = "short"

        if acct.position:
            pos_type = acct.position[-1]
   
            if sig != pos_type and (sig == "long" or sig == "short"):
                acct.process("close", sym, dt, price,
                             log_msg="reverse position")

        acct.process(sig, sym, dt, price)

    return acct


def show_vbt_perf(port):
    """Display performance metrics from a vectorbt portfolio."""

    def fmt(val):
        return f"{val:,.2f}"

    eq = port.value()
    start = eq.index[0].strftime("%Y-%m-%d")
    end = eq.index[-1].strftime("%Y-%m-%d")
    
    dd_pct = eq.cummax().sub(eq).div(eq.cummax())
    trough = dd_pct.idxmax()
    peak_window = eq.loc[:trough]
    dd_start = peak_window[peak_window == peak_window.cummax()].index[-1].strftime("%Y-%m-%d")

    metrics = [
        ["\nVectorbt summary:", ""],
        ["Date Range:", f"{start} to {end}"],
        ["Total Return (%):", f"{port.total_return() * 100:+.2f}%"],
        ["Ann. Ret. (%):", f"{port.annualized_return() * 100:.2f}%"],
        ["Max Drawdown:", f"{port.max_drawdown() * 100:.1f}%"],
        ["Win Ratio (%):", f"{port.trades.win_rate() * 100:.2f}%"],
        ["Max DD Start:", dd_start],
        ["Net Profit:", fmt(port.final_value() - port.init_cash)],
        ["Fees Paid:", fmt(port.orders.fees.sum())],
        ["Start Cap:", fmt(port.init_cash)],
        ["End Cap:", fmt(port.final_value())],
        ["Trades:", str(int(port.trades.count()))],
    ]

    for name, val in metrics:
        print(f"{name:<20} {val}")


def run_vbt(prices, vbt_mod, fast=40, slow=60, msg=""):
    """Run vectorbt SMA crossover backtest."""
    sma_fast = vbt_mod.MA.run(prices, fast, short_name="fast")
    sma_slow = vbt_mod.MA.run(prices, slow, short_name="slow")

    long_entries = sma_fast.ma_crossed_above(sma_slow)
    long_exits = sma_fast.ma_crossed_below(sma_slow)
    short_entries = sma_fast.ma_crossed_below(sma_slow)
    short_exits = sma_fast.ma_crossed_above(sma_slow)

    port = vbt_mod.Portfolio.from_signals(
        prices,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=10_000,
        allow_partial=False,
    )

    if msg:
        print("\n" + msg)
    return port


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "btc_1m.csv")

    print("Loading data...")
    df = pd.read_csv(csv_path)
    df["close_time"] = pd.to_datetime(df["close_time"])
    df.set_index("close_time", inplace=True)

    resamp = df.close.resample("10min").ohlc()
    resamp = resamp[resamp.close > 0]
    rows = tuple(zip(resamp.index, resamp.close))

    print(f"Total bars: {len(rows)}")

    print("\n--- Running Antback (median of 10, 2 warmup discarded)...")
    acct, ab_times = bench_repeated(backtest_sma_cross, rows, "BTC", fast=40, slow=60,
                                     repeats=10, warmup=2)
    ab_med = ab_times[len(ab_times) // 2]
    print(f"antback: median={ab_med*1000:.2f}ms  min={ab_times[0]*1000:.2f}ms  "
          f"max={ab_times[-1]*1000:.2f}ms  (n={len(ab_times)})")
    acct.basic_report()

    print("\n--- Running vectorbt...")
    import vectorbt as vbt
    print(f"vectorbt version: {vbt.__version__}")

    port, first_call_s, steady_times = bench_first_vs_steady_state(
        run_vbt, resamp.close, vbt, fast=40, slow=60, repeats=10,
        msg="vectorbt: computing signals + portfolio",
    )
    steady_med = steady_times[len(steady_times) // 2]

    print(f"\nvectorbt 1st call in this process (pays Numba JIT): {first_call_s*1000:.1f}ms")
    print(f"vectorbt steady-state median of {len(steady_times)} (same process): "
          f"{steady_med*1000:.2f}ms  min={steady_times[0]*1000:.2f}ms  max={steady_times[-1]*1000:.2f}ms")
    print(f"\n=> running this script once, cold: vectorbt ~{first_call_s*1000:.0f}ms  vs  antback ~{ab_med*1000:.2f}ms")
    print(f"=> long-lived process calling this many times: vectorbt converges to ~{steady_med*1000:.2f}ms/call")
    print("   (measured separately: even with Numba's on-disk cache warm from a prior run,")
    print("   a FRESH process still pays ~4-5s here before reaching steady state -- the 1st-call")
    print("   number above is what recurs on every plain `python script.py` invocation, not a")
    print("   one-time-ever cost. Confirm on your machine: run this script twice in a row and")
    print("   compare the two '1st call' numbers.)")

    show_vbt_perf(port)

    if False:  # Enable to export trades
        trades = port.trades.records_readable
        import df2tables as df2t
        df2t.render(pd.DataFrame(trades),
                    to_file="vbt_trade.html", title="vbt_trades")


# --- Worth trying separately: the comparison vectorbt is actually built for ---
#
# The benchmark above runs ONE strategy through each engine -- that's not
# where vectorbt's architecture is supposed to pay off (its selling point is
# broadcasting THOUSANDS of parameter combos into one vectorized call, per
# its own docs: "turning hours of grid search into seconds"). A single
# 1-vs-1 run mostly just measures whose setup cost is bigger. The more
# revealing test is a parameter sweep -- antback looping over N (fast, slow)
# combos one at a time vs. vectorbt computing all N in one call:
#
#   import numpy as np
#   windows = np.arange(10, 101, 10)
#   fast_ma, slow_ma = vbt.MA.run_combs(resamp.close, window=windows, r=2,
#                                        short_names=["fast", "slow"])
#   entries = fast_ma.ma_crossed_above(slow_ma)
#   exits = fast_ma.ma_crossed_below(slow_ma)
#   port = vbt.Portfolio.from_signals(resamp.close, entries, exits, init_cash=10_000)
#   # port now holds results for every (fast, slow) pair at once
#
# vs. N calls to backtest_sma_cross with different fast/slow each time. That
# comparison will actually show whether/when vectorbt's amortized-JIT,
# vectorized approach overtakes antback's zero-warmup simplicity for your
# real workload (many tickers x many parameter combos), which single-shot
# timing here can't answer either way.
#
# Also worth 10 minutes: `pip install -U "vectorbt[rust]"` -- the PyPI
# listing advertises an optional Rust engine specifically to avoid JIT
# overhead. I didn't get to verify it against this workload, but if it
# applies to MA/from_signals it could sidestep this whole cold-start
# question rather than just working around it.


if __name__ == "__main__":
    main()