import time
import pandas as pd
import antback as ab

def timeit(func):
    """Decorator to measure execution time of a function."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"'{func.__name__}' executed in: {end - start:.4f}s")
        return result

    return wrapper


@timeit
def backtest_sma_cross(rows, sym, fast=5, slow=50):
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
            if sig != pos_type and sig == "long" or sig == "short":
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

    metrics = [
        ["\nVectorbt summary:", ""],
        ["Date Range:", f"{start} to {end}"],
        ["Total Return (%):", f"{port.total_return() * 100:+.2f}%"],
        ["Ann. Ret. (%):", f"{port.annualized_return() * 100:.2f}%"],
        ["Max Drawdown:", f"{port.max_drawdown() * 100:.1f}%"],
        ["Win Ratio (%):", f"{port.trades.win_rate() * 100:.2f}%"],
        [
            "Max DD Start:",
            eq.cummax().sub(eq).div(eq.cummax()).idxmin().strftime("%Y-%m-%d"),
        ],
        ["Net Profit:", fmt(port.final_value() - port.init_cash)],
        ["Fees Paid:", fmt(port.orders.fees.sum())],
        ["Start Cap:", fmt(port.init_cash)],
        ["End Cap:", fmt(port.final_value())],
        ["Trades:", str(int(port.trades.count()))],
    ]

    for name, val in metrics:
        print(f"{name:<20} {val}")


@timeit
def run_vbt(prices, vbt_mod, fast=10, slow=30, msg=""):
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

    print("\n" + msg)
    return port


def main():
    print("Loading data...")
    df = pd.read_csv("btc_1m.csv")
    df["close_time"] = pd.to_datetime(df["close_time"])
    df.set_index("close_time", inplace=True)

    resamp = df.close.resample("10min").ohlc()
    rows = tuple(zip(resamp.index, resamp.close))

    print(f"Total bars: {len(rows)}")

    print("\n--- Running Antback...")
    acct = backtest_sma_cross(rows, "BTC", fast=40, slow=60)
    acct.basic_report()
    print("\n--- Running vectorbt...")
    
    # Import vectorbt only after Antback runs.
    # The import takes a little time, so we defer it 
    # to keep Antback timing clean. We then pass `vectorbt` 
    # as `vbt_mod` (dependency injection).
    
    import vectorbt as vbt
    
    print(f"vectorbt version: {vbt.__version__}") #vectorbt version:  0.28.1
    _ = run_vbt(resamp.close, vbt, fast=40, slow=60, msg="vectorbt: compilation run")
    port = run_vbt(resamp.close, vbt, fast=40, slow=60, msg="vectorbt: compiled run")

    show_vbt_perf(port)

    if False:  # Enable to export trades
        trades = port.trades.records_readable
        import df2tables as df2t
        df2t.render(pd.DataFrame(trades),
                    to_file="vbt_trade.html", title="vbt_trades")



if __name__ == "__main__":
    main()
