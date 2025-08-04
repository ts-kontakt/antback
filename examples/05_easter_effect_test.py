import pandas as pd
from dateutil.easter import easter
import antback as ab


def get_easter_dates(price_df):
    """
    Generate Easter Sunday dates for each unique year in the DataFrame index.

    Args:
        price_df (pd.DataFrame): DataFrame with datetime index.

    Returns:
        list of pd.Timestamp: Timezone-aware (if applicable) Easter dates.
    """
    return [
        pd.Timestamp(easter(year)).tz_localize(price_df.index.tz) 
        if price_df.index.tz else pd.Timestamp(easter(year))
        for year in price_df.index.year.unique()
    ]


def get_july4_dates(price_df):
    """
    Generate July 4th dates for each unique year in the DataFrame index.

    Args:
        price_df (pd.DataFrame): DataFrame with datetime index.

    Returns:
        list of pd.Timestamp: Timezone-aware (if applicable) July 4th dates.
    """
    return [
        pd.Timestamp(year, 7, 4).tz_localize(price_df.index.tz) 
        if price_df.index.tz else pd.Timestamp(year, 7, 4)
        for year in price_df.index.year.unique()
    ]


def get_pre_holiday_trading_dates(price_df, holiday_func, days_before=1):
    """
    Determine trading dates a specified number of days before each holiday.

    Args:
        price_df (pd.DataFrame): Market data with datetime index.
        holiday_func (function): Function that returns a list of holiday pd.Timestamps.
        days_before (int): Number of trading days before each holiday.

    Returns:
        list of pd.Timestamp: Trading dates before the holidays.
    """
    return ab.get_pre_dates_for_targets(price_df, holiday_func(price_df), days_before)


def test_holiday_effect(price_df, ticker):
    """
    Simulate a holiday-based trading strategy.
    Buy one day before Easter, sell after holding for 4 bars (days).

    The strategy buys at the close price on the day before the holiday
    and sells at the open price after a fixed holding period.

    Args:
        price_df (pd.DataFrame): Historical price data with 'Open' and 'Close'.
        ticker (str): Asset ticker symbol.

    Returns:
        ab.Portfolio: Portfolio object with full report of strategy performance.
    """
    portfolio = ab.Portfolio(cash=10_000, single=True)
    buy_signals = get_pre_holiday_trading_dates(price_df, get_easter_dates, days_before=1)
    sell_timer = ab.new_wait_n_bars(4)  # Exit after 4 bars

    for row in price_df.to_records():
        current_date, open_price, close_price = row
        current_date = pd.to_datetime(current_date)
        signal = None

        if sell_timer(bar=current_date):
            signal = 'sell'
        elif current_date in buy_signals:
            signal = 'buy'
            sell_timer(start=True)

        trade_price = open_price if signal == 'sell' else close_price
        portfolio.process(signal, ticker, current_date, trade_price)

    portfolio.full_report(title=f'Holiday Effect Strategy: {ticker}')
    return portfolio


if __name__ == "__main__":
    import yfinance as yf

    TICKER = "QQQ"  # Nasdaq-100 ETF
    price_data = yf.Ticker(TICKER).history(period='10y')[['Open', 'Close']]

    test_holiday_effect(price_data, TICKER)
