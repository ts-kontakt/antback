from talipp.indicators import SMA
import antback as ab


def test_sma_crossover(data, symbol, fast=10, slow=30):
    # Initialize portfolio and indicators
    port = ab.Portfolio(10_000, single=True)
    cross = ab.new_cross_func()
    fast_sma, slow_sma = SMA(period=fast), SMA(period=slow)

    for date, price in data.items():  # For dataframe use .to_records()
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

    # Generate performance reports
    port.basic_report()
    port.full_report(
        "html",
        outfile="indicators.html",
        title=f"SMA Crossover ({fast}/{slow}) on {symbol}",
    )


def main():
    import yfinance as yf

    symbol = "QQQ"
    data = yf.Ticker(symbol).history(period="10y")["Close"]
    test_sma_crossover(data, symbol)


if __name__ == "__main__":
    main()
