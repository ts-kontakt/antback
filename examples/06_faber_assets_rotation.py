from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import antback as ab


def fetch_price_history(tickers, years_back=10):
    """
    Fetch historical daily closing prices for the given assets.

    This is used as input for the Mebane Faber-inspired 
    moving-average rotation strategy.

    Args:
        tickers (list[str]): List of asset tickers (e.g., ['SPY', 'GLD']).
        years_back (int, optional): Number of years of historical data to fetch.
            Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame with daily closing prices for the specified tickers.
    """
    assert isinstance(tickers, list), "tickers must be a list of strings"
    end_date = date.today()
    start_date = end_date - timedelta(days=years_back * 365)

    price_history = yf.download(tickers, start=start_date, end=end_date)["Close"]
    return price_history


def run_backtest(price_history, tickers, rolling_months=10):
    """
    Run a simple moving average (SMA) crossover backtest.

    Strategy Origin:
        Based on Mebane Faber's published "Ivy Portfolio" methodology
        for tactical asset allocation.  
        - Compare a short-term SMA (3 months) with a long-term SMA (10 months).
        - Stay invested in assets that are trending up, and rotate out of those
          trending down.
        - Rebalance at month-end.

    Important Implementation Note:
        We do **not** resample to monthly prices to calculate the 10-month SMA.
        Instead, we keep daily price data in rolling lists, but only make trading
        decisions at month-end. The list of month-end dates comes from 
        `antback.get_monthly_points()`, which avoids any unintended averaging effects
        from using monthly bars.

    Args:
        price_history (pd.DataFrame): Historical daily closing prices.
        tickers (list[str]): List of asset tickers to backtest.
        rolling_months (int, optional): Rolling period length in months for SMA calculation.
            Defaults to 10.

    Returns:
        ab.Portfolio: Portfolio object containing performance results.
    """
    # Lower transaction fees since portfolio implementation forces full reallocation
    transaction_fee = 0.0015 / 2  

    portfolio = ab.Portfolio(
        cash=100_000,
        single=False,
        fees=transaction_fee,
        allow_fractional=False
    )

    # Identify month-end dates for decision points
    month_end_dates = ab.get_monthly_points(price_history)["end"]

    # Rolling price history storage for SMA calculations
    rolling_prices = ab.NamedRollingLists(rolling_months)

    # Iterate over all price records
    for record in price_history.to_records():
        current_date = pd.to_datetime(record[0])
        prices = {ticker: float(getattr(record, ticker)) for ticker in tickers}

        # Skip if any asset price is missing for the current date
        if np.any(pd.isna(list(prices.values()))):
            continue

        # Execute trades only at month-end
        if current_date in month_end_dates:
            # Append latest prices to rolling storage
            for ticker, price in prices.items():
                rolling_prices.append(ticker, price)

            # Only act when we have enough data for SMA calculation
            if len(rolling_prices.get(tickers[0])) == rolling_months:
                buy_list, sell_list = [], []

                for ticker in tickers:
                    history = rolling_prices.get(ticker)
                    sma_long = np.mean(history)       # 10-month SMA
                    sma_short = np.mean(history[-3:]) # 3-month SMA

                    if sma_short > sma_long:
                        buy_list.append(ticker)
                    else:
                        sell_list.append(ticker)

                # Compare current positions with desired positions
                current_positions = tuple(portfolio.positions.keys())
                position_changes = set(buy_list).symmetric_difference(current_positions)

                if position_changes:
                    print(current_positions, 'diff:', position_changes, 'buy:', buy_list, '- sell:', sell_list)

                    # Sell all current positions before buying
                    for ticker in current_positions:
                        portfolio.sell(ticker, current_date, prices[ticker])

                    # Allocate equally among buy_list
                    if buy_list:
                        allocation_per_asset = portfolio.cash / len(buy_list)
                        for ticker in buy_list:
                            portfolio.buy(ticker, current_date, prices[ticker], fixed_val=allocation_per_asset)
                else:
                    print('- No changes', current_positions, buy_list, sell_list)

        # Update portfolio market values daily
        for ticker in tickers:
            portfolio.update(ticker, current_date, prices[ticker])

    return portfolio


def main():
    """
    Entry point for the SMA crossover backtest script.

    This backtest implements a variation of the Mebane Faber
    tactical asset rotation strategy described in "The Ivy Portfolio".

    Unlike some implementations, it does not use monthly bars for SMA calculation.
    Daily data is preserved, and month-end decision points are determined via 
    `antback.get_monthly_points()`.
    """
    ROLLING_AVG_MONTHS = 10  # Number of months for the long-term moving average
    tickers = ['SPY', 'GLD', 'TLT']

    # Optional: Uncomment to include Bitcoin
    # tickers.append('BTC-USD')

    # Fetch data and execute backtest
    price_history = fetch_price_history(tickers, years_back=10)
    backtest_results = run_backtest(price_history, tickers, ROLLING_AVG_MONTHS)

    # Display performance reports
    backtest_results.basic_report()
    backtest_results.full_report()
    # backtest_results.full_report('excel')


if __name__ == "__main__":
    main()
