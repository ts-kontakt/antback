import antback as ab


def turn_of_the_month_strategy(price_series, ticker):
    """
    Simulate a simple 'Turn of the Month' trading strategy.

    The strategy buys near the end of the month and sells near the beginning of the next.
    `near_start_day` and `near_end_day` define the range of days considered "near" the start/end.

    Args:
        price_series (pd.Series): Daily closing prices with datetime index.
        ticker (str): Ticker symbol of the asset being traded..
    """
    portfolio = ab.Portfolio(cash=10_000, single=True)

    # Identify calendar-based trading points (customizable)
    monthly_timing = ab.get_monthly_points(
        price_series, near_start_day=3, near_end_day=3
    )

    for current_date, close_price in price_series.items():
        trade_signal = None
        # Buy near end of the month
        if current_date in monthly_timing["near_end"]:
            trade_signal = "buy"

        # Sell near start of the next month
        if current_date in monthly_timing["near_start"]:
            trade_signal = "sell"

        # Process trade only if a signal exists
        portfolio.process(trade_signal, ticker, current_date, close_price)

    portfolio.basic_report()
    portfolio.full_report(title="Turn of the Month Strategy")

    return portfolio


def main():
    import yfinance as yf

    TICKER = "QQQ"  # Nasdaq-100 ETF
    price_data = yf.Ticker(TICKER).history(period="10y")

    # Run the strategy on closing prices
    turn_of_the_month_strategy(price_data["Close"], TICKER)


if __name__ == "__main__":
    main()
