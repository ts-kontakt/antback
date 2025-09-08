import time
import pandas as pd

import antback as ab


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"'{func.__name__}' executed in: {end - start:.4f}s")
        return result

    return wrapper


@timer
def backtest_sma_cross(prices, symbol, fast=5, slow=50):
    """Backtest SMA crossover strategy using Antback backtesting engine."""
    account = ab.CFDAccount(
        cash=10_000, margin_requirement=0.1, leverage=1, fees=0.0, warn=0
    )
    cross_func = ab.new_cross_func()
    sma_fast = ab.create_sma_func(period=fast)
    sma_slow = ab.create_sma_func(period=slow)

    for timestamp, price in prices:
        signal = "update"
        fast_val = sma_fast(price)
        slow_val = sma_slow(price)

        if fast_val and slow_val:
            direction = cross_func(fast_val, slow_val)
            if direction == "up":
                signal = "long"
            elif direction == "down":
                signal = "short"

        if account.position:
            position_type = account.position[-1]
            if signal != position_type and signal in ("long", "short"):
                account.process(
                    "close", symbol, timestamp, price, log_msg="reverse position"
                )

        account.process(signal, symbol, timestamp, price)

    return account


def format_currency(value):
    """Format number as currency string."""
    return f"{value:,.2f}"


def show_vbt_performance(portfolio):
    """Display performance metrics from a vectorbt portfolio."""
    equity = portfolio.value()
    start_date = equity.index[0].strftime("%Y-%m-%d")
    end_date = equity.index[-1].strftime("%Y-%m-%d")

    metrics = [
        ["Vectorbt portfolio summary:", ""],
        ["Date Range:", f"{start_date} to {end_date}"],
        ["Total Return (%):", f"{portfolio.total_return() * 100:+.2f}%"],
        ["Ann. Ret. (%):", f"{portfolio.annualized_return() * 100:.2f}%"],
        ["Max Drawdown:", f"{portfolio.max_drawdown() * 100:.1f}%"],
        ["Winning Ratio (%):", f"{portfolio.trades.win_rate() * 100:.2f}%"],
        [
            "Max DD Start:",
            equity.cummax().sub(equity).div(equity.cummax()).idxmin().strftime("%Y-%m-%d"),
        ],
        ["Net Profit:", format_currency(portfolio.final_value() - portfolio.init_cash)],
        ["Total Fees Paid:", format_currency(portfolio.orders.fees.sum())],
        ["Starting Capital:", format_currency(portfolio.init_cash)],
        ["Ending Capital:", format_currency(portfolio.final_value())],
        ["Number of Trades:", str(int(portfolio.trades.count()))],
    ]

    for name, value in metrics:
        print(f"{name:<20} {value}")

import vectorbt as vbt
@timer
def run_vbt(prices, fast=10, slow=30, message=""):
    """Run vectorbt MA crossover backtest."""
    fast_ma = vbt.MA.run(prices, fast, short_name="fast")
    slow_ma = vbt.MA.run(prices, slow, short_name="slow")

    long_entries = fast_ma.ma_crossed_above(slow_ma)
    long_exits = fast_ma.ma_crossed_below(slow_ma)
    short_entries = fast_ma.ma_crossed_below(slow_ma)
    short_exits = fast_ma.ma_crossed_above(slow_ma)

    portfolio = vbt.Portfolio.from_signals(
        prices,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=10_000,
        allow_partial=False,
    )

    print('\n' + message)
    return portfolio


def main():
    print("Loading data...")
    df = pd.read_csv("btc_1m.csv")
    df["close_time"] = pd.to_datetime(df["close_time"])
    df.set_index("close_time", inplace=True)
    resampled = df.close.resample("10min").ohlc()
    historical_prices = tuple(zip(resampled.index, resampled.close))

    print(f"Total bars: {len(historical_prices)}")

    print("\n--- Running antback...")
    custom_result = backtest_sma_cross(historical_prices, "BTC", fast=40, slow=60)
    custom_result.basic_report()

    print("\n--- Running vectorbt...")
    _ = run_vbt(resampled.close, fast=40, slow=60, message="vectorbt: compilation run")
    vbt_result = run_vbt(resampled.close, fast=40, slow=60, message="vectorbt: compiled run")

    show_vbt_performance(vbt_result)

    if False:  # Set to True to enable trade display
        trades = vbt_result.trades.records_readable
        import df2tables as df2t
        df2t.render(pd.DataFrame(trades), to_file="vbt_trade.html", title="vbt_trades")


if __name__ == "__main__":
    main()
