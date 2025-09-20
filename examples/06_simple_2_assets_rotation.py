from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

import antback as ab



def get_price_data(primary_ticker, alternative_ticker, years_back=10):
    """
    Fetch historical daily closing prices for both primary and alternative assets
    over a specified number of years.

    Args:
        primary_ticker (str): Ticker symbol of the primary asset.
        alternative_ticker (str): Ticker symbol of the alternative asset.
        years_back (int, optional): Number of years of historical data to fetch. Defaults to 10.

    Returns:
        pd.DataFrame: Daily closing prices for the selected tickers.
    """
    tickers = [alternative_ticker, primary_ticker]
    end_date = date.today()
    start_date = end_date - timedelta(days=years_back * 365)

    price_data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    # price_data = price_data.fillna(method='ffill')
    return price_data


def run_backtest(
    price_data,
    primary_ticker,
    alternative_ticker,
    rolling_months=10,
    decision_point="end",
):
    """
    Execute a tactical asset allocation backtest using a simple rolling-average signal.
    Switch between the primary and alternative assets based on decision points
    (end, near_end, mid, near_start, etc.).

    Thanks to the built-in `get_monthly_points` function, decisions can be simulated
    at specific monthly intervals without resampling to monthly bars. This preserves
    daily price fidelity for accurate portfolio tracking.

    Args:
        price_data (pd.DataFrame): Daily price data for both assets.
        primary_ticker (str): Ticker of the primary asset (e.g., SPY).
        alternative_ticker (str): Ticker of the alternative asset (e.g., GLD).
        rolling_months (int, optional): Number of months to use for rolling average signal. Defaults to 10.
        decision_point (str, optional): When to make allocation decisions.
            Options: "end", "near_end", "mid", "near_start". Defaults to "end".

    Returns:
        ab.Portfolio: The portfolio object containing performance results.
    """
    portfolio = ab.Portfolio(cash=100_000, single=True, warn=False)

    # Identify monthly decision dates
    monthly_decision_dates = ab.get_monthly_points(price_data, near_start_day=3,
                                                   near_end_day=3)[decision_point]

    rolling_window = ab.RollingList(rolling_months)

    for row in price_data.itertuples():
        current_date, alt_price, primary_price = row
        current_date = pd.to_datetime(current_date)

        # Trade only on chosen monthly decision dates
        if current_date in monthly_decision_dates:
            rolling_window.append(primary_price)
            historical_prices = rolling_window.values()

            if len(historical_prices) == rolling_months:
                avg_primary_price = np.nanmean(historical_prices)
                if primary_price > avg_primary_price:
                    # Allocate to primary asset if above average
                    portfolio.sell(alternative_ticker, current_date, alt_price)
                    portfolio.buy(primary_ticker, current_date, primary_price)
                else:
                    # Allocate to alternative asset if below average
                    portfolio.sell(primary_ticker, current_date, primary_price)
                    portfolio.buy(alternative_ticker, current_date, alt_price)

        # Always update mark-to-market values
        portfolio.update(primary_ticker, current_date, primary_price)
        portfolio.update(alternative_ticker, current_date, alt_price)

    return portfolio


def main():
    # Define assets for tactical allocation
    primary_ticker = "SPY"
    alternative_ticker = "GLD"  # Gold ETF
    rolling_avg_months = 10  # Use 10-month rolling average
    decision_point = "end"  # Can be "end", "near_end", "mid", or "near_start"

    # Fetch historical data and run backtest
    price_data = get_price_data(primary_ticker, alternative_ticker, years_back=20)
    backtest_results = run_backtest(
        price_data,
        primary_ticker,
        alternative_ticker,
        rolling_months=rolling_avg_months,
        decision_point=decision_point,
    )

    # Output performance summary
    backtest_results.basic_report()
    backtest_results.full_report()


if __name__ == "__main__":
    main()
