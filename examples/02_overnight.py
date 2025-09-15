import antback as ab


def overnight_strategy(df, ticker: str, entry_day: int, hold_days: int = 1):
    """
    Simulate an overnight trading strategy:
    - Buy at the close on a specific weekday.
    - Hold for a set number of trading days.
    - Sell at the open after the holding period.

    Notes:
        • This strategy explores the so-called "weekday overnight effect,"
          where returns can vary significantly depending on which weekday
          positions are opened and how many days they are held.
        • Some entry/exit combinations historically show stronger patterns
          (e.g., buying Monday close and selling mid-week), while others may
          perform poorly (e.g., holding across weekends).
        • Transaction costs are ignored here (fees=0) to focus on the raw
          overnight effect. In practice, fees would often make the strategy
          ** unprofitable ** .

    Args:
        df (pd.DataFrame): OHLC price data with datetime index.
        ticker (str): Ticker symbol (e.g., "QQQ").
        entry_day (int): Weekday to buy (0=Monday ... 4=Friday).
        hold_days (int, optional): Number of trading days to hold before selling.
                                   Defaults to 1 (buy close → sell next open).

    Returns:
        ab.Portfolio: Portfolio object containing trades and performance reports.
    """
    assert entry_day in [0, 1, 2, 3, 4], "entry_day must be between 0 (Mon) and 4 (Fri)."

    portfolio = ab.Portfolio(cash=10_000, single=True, fees=0)

    sell_timer = ab.new_wait_n_bars(hold_days)
    for row in df.itertuples():
        date, open_price, high, low, close_price = row[:5]
        signal = None

        # Exit trade if the holding period is over
        if sell_timer(bar=date):
            signal = "sell"

        # Enter trade if it's the chosen weekday
        if date.weekday() == entry_day:
            signal = "buy"
            sell_timer(start=True)

        # Buy at close, sell at open
        trade_price = close_price if signal == "buy" else open_price
        portfolio.process(signal, ticker, date, trade_price)

    portfolio.basic_report()
    portfolio.full_report(title="Overnight Strategy")

    return portfolio


def main():
    import yfinance as yf

    ticker = "QQQ"  # Nasdaq-100 ETF
    price_data = yf.Ticker(ticker).history(period="10y")

    ## Example: Buy Monday close → Sell Tuesday open (1-day hold)
    overnight_strategy(price_data, ticker, entry_day=0, hold_days=1)

    ## Example: Buy Monday close → Sell Wednesday open (2-day hold)
    # overnight_strategy(price_data, ticker, entry_day=0, hold_days=2)

    ## Example: Buy Friday close → Sell Monday open (1-day hold)
    # overnight_strategy(price_data, ticker, entry_day=4, hold_days=1)


if __name__ == "__main__":
    main()
