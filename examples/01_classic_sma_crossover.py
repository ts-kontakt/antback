import numpy as np

import antback as ab


def test_simple_sma(data, symbol, fast=10, slow=30):
    # Initialize portfolio and data storage
    port = ab.Portfolio(10_000, single=True)
    prices = ab.RollingList(maxlen=slow)
    cross = ab.new_cross_func()

    for date, price in data.items():
        prices.append(price)
        price_history = prices.values()
        signal = "update"  # Reset signal for each iteration
        if len(price_history) >= slow:
            
            fast_ma = np.mean(price_history[-fast:])
            slow_ma = np.mean(price_history[-slow:])
            direction = cross(fast_ma, slow_ma)

            if direction == "up":
                signal = "buy"
            elif direction == "down":
                signal = "sell"
        port.process(signal, symbol, date, price)

    # Generate reports
    port.basic_report()
    port.full_report(
        "html",
        outfile=f"sma_crossover_report.html",
        title=f"Simple SMA Crossover on {symbol}",
    )



def main():
    import yfinance as yf

    symbol = "QQQ"
    data = yf.Ticker(symbol).history(period="10y")
    test_simple_sma(data.Close, symbol)


if __name__ == "__main__":
    main()
