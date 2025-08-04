#!/usr/bin/python
# coding=utf-8
import os
import sys
from collections import defaultdict, namedtuple
from datetime import date, datetime, timedelta
from pprint import pprint

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

try:
    from . import utils
    from .comnt import write_from_template
except ImportError:
    import utils
    from comnt import write_from_template


def pct_diff(prev, today):
    prev, today = float(prev), float(today)
    try:
        return (today - prev) / prev * 100
    except ZeroDivisionError:
        return 0.0


TradeData = namedtuple(
    "TradeData",
    [
        "ticker",
        "buy_date",
        "buy_price",
        "sell_date",
        "sell_price",
        "quantity",
        "log_msg",
    ],
)

TradeResult = namedtuple(
    "TradeResult",
    [
        "ticker",
        "buy_date",
        "sell_date",
        "buy_price",
        "sell_price",
        "quantity",
        "position_size",
        "net_profit",
        "profit_pct",
        "gross_profit",
        "buy_fees",
        "sell_fees",
        "total_fees",
        "log_msg",
    ],
)


def trade_result(trade, brokerage, opened=False):
    """Calculate trade results including comissions."""
    position_size = trade.buy_price * trade.quantity
    sell_value = trade.sell_price * trade.quantity
    buy_fees = position_size * brokerage
    if not opened:
        sell_fees = sell_value * brokerage
        total_fees = buy_fees + sell_fees
    else:
        sell_fees = 0
        total_fees = buy_fees

    profit_pct = pct_diff(trade.buy_price, trade.sell_price)
    gross_profit = sell_value - position_size
    net_profit = gross_profit - total_fees

    log_msg = "*opened" if opened else trade.log_msg

    return TradeResult(
        ticker=trade.ticker,
        buy_date=trade.buy_date,
        sell_date=trade.sell_date,
        buy_price=trade.buy_price,
        sell_price=trade.sell_price,
        quantity=trade.quantity,
        position_size=position_size,
        net_profit=net_profit,
        profit_pct=profit_pct,
        gross_profit=gross_profit,
        buy_fees=buy_fees,
        sell_fees=sell_fees,
        total_fees=total_fees,
        log_msg=log_msg,
    )



class Portfolio:
    """Trading portfolio management class."""
    DEFAULT_PERCENT = 1.0
    MIN_AMOUNT = 500
    def __init__(
        self, cash, single=True, warn=False, allow_fractional=False, fees=0.0015
    ):
        self.warn = warn
        assert cash >= 10000, "Minimum cash requirement: 10,000"
        self.starting_capital = cash
        self.cash = cash
        self.positions = {}
        self.fees = fees
        self.max_date = None
        self.trades = []
        self.pos_history = {}
        self.single = single
        self.events = defaultdict(list)
        self.total_fees_paid = 0.0
        self.allow_fractional = allow_fractional

    def _normalize_date(self, date_obj):
        """Normalize date object to datetime/date."""
        if isinstance(date_obj, (date, datetime)):
            return date_obj
        return pd.to_datetime(date_obj)

    def _validate_and_update_date(self, date_obj):
        """Validate date format and update max_date."""
        date_obj = self._normalize_date(date_obj)
        assert isinstance(date_obj, (date, datetime)), (
            "date_obj must be date or datetime"
        )

        if self.max_date and date_obj < self.max_date:
            print(f"! Recived date: {date_obj}, self.max_date: {self.max_date}")
            raise ValueError("Date not in chronological order")

        if not self.max_date or date_obj > self.max_date:
            self.max_date = date_obj

        return date_obj

    def buy(self, ticker, date_obj, price, fixed_val=None):
        """Buy shares of a ticker."""
        if fixed_val is None:
            if not self.single:
                raise AssertionError(
                    "For multi ticker portfolio fixed_val buy value is expected"
                )

        date_obj = self._validate_and_update_date(date_obj)
        price = float(price)

        if self.single and self.positions and ticker not in self.positions:
            raise Exception(f"Single mode - can't add new ticker: {ticker}")

        if ticker in self.positions:
            if self.warn:
                print(f"! Re-buying not allowed: {ticker} {date_obj}")
            return

        if fixed_val:
            assert price < fixed_val
            assert fixed_val >= self.MIN_AMOUNT, f"fixed_val amount must be >= {
                self.MIN_AMOUNT
            }"
            if self.cash < fixed_val:
                msg = f"Insufficient cash: {self.cash:.1f} < {fixed_val}"
                self.events[date_obj].append(msg)
                raise ValueError(msg)
            available_investment = fixed_val
        else:
            available_investment = self.cash * self.DEFAULT_PERCENT #1.0

        buffer_factor = 0.99999  # Prevent error comparing floats
        max_affordable_shares = (available_investment * buffer_factor) / (
            price * (1 + self.fees)
        )

        quantity = (
            max_affordable_shares
            if self.allow_fractional
            else int(max_affordable_shares)
        )

        if quantity <= 0.001:
            msg = f"""! buy quantity <= 0.001 for {ticker} at {price}: need {
                price * (1 + self.fees)},
             have {available_investment:.2f}"""
            self.events[date_obj].append(msg)
            if self.warn:
                print(msg)
            return

        share_cost = price * quantity
        fees = share_cost * self.fees
        total_cost = share_cost + fees

        if self.cash < total_cost:
            if self.warn:
                raise ValueError(
                    f"Calculation error: cash {self.cash:.5f} < needed {total_cost:.5f}"
                )

        self.cash -= total_cost
        self.total_fees_paid += fees
        self.positions[ticker] = [date_obj, quantity, price, date_obj, price]

        self.events[date_obj].append(
            f"buy: {ticker}({price:.2f}*{quantity:.1f}, cost: {total_cost:.1f}, fees: {fees:.1f})"
        )

        if self.warn:
            print(
                f"""Bought {quantity:.4f} shares of {ticker} at {price:.2f}, total cost: {
                    total_cost:.2f
                }"""
            )

        self.save_positions(date_obj)

    def sell(self, ticker, date_obj, price, log_msg=""):
        """Sell shares of a ticker."""
        date_obj = self._validate_and_update_date(date_obj)
        price = float(price)

        if ticker not in self.positions:
            if self.warn:
                print(f"Ticker not in portfolio: {ticker}")
            return

        buy_date, quantity, buy_price, last_date, last_price = self.positions.pop(
            ticker
        )

        gross_proceeds = price * quantity
        fees = gross_proceeds * self.fees
        net_proceeds = gross_proceeds - fees

        self.trades.append(
            TradeData(ticker, buy_date, buy_price, date_obj, price, quantity, log_msg)
        )

        self.cash += net_proceeds
        self.total_fees_paid += fees

        self.events[date_obj].append(
            f"sell: {ticker}({price:.2f}*{quantity:.1f}, proceeds: {net_proceeds:.1f}, fees: {fees:.1f})"
        )

        if self.warn:
            print(
                f"""Sold {quantity:.4f} shares of {ticker} at {price:.2f}, net proceeds: {
                    net_proceeds:.2f
                }"""
            )

        self.save_positions(date_obj)

    def update(self, ticker, date_obj, price):
        """Update price for existing position."""
        date_obj = self._validate_and_update_date(date_obj)
        price = float(price)
        assert price > 0, "Price must be positive"

        if ticker in self.positions:
            self.positions[ticker][-2] = date_obj
            self.positions[ticker][-1] = price

        self.save_positions(date_obj)

    def process(self, signal, ticker, date_obj, price, buy_fixed=None, log_msg=""):
        """Unified entry-point for any daily signal."""
        if signal is None:
            self.update(ticker, date_obj, price)
        elif signal == "buy":
            self.buy(ticker, date_obj, price, fixed_val=buy_fixed)
        elif signal == "sell":
            self.sell(ticker, date_obj, price, log_msg=log_msg)
        else:
            raise ValueError(f"Unknown signal {signal!r}")

    def save_positions(self, record_date):
        """Save current portfolio state."""
        pos_list = []
        for ticker, (
            buy_date,
            qty,
            buy_price,
            last_date,
            last_price,
        ) in self.positions.items():
            pos_list.append((ticker, qty, last_price))
        self.pos_history[record_date] = (self.cash, pos_list)

    def get_open_positions(self):
        """Get information about open positions."""
        if not self.positions:
            return {"positions_total": 0, "open_trades": []}

        open_trades = []
        total = 0

        for ticker, (
            buy_date,
            quantity,
            buy_price,
            last_date,
            last_price,
        ) in self.positions.items():
            if not last_date:
                last_date = self.max_date or datetime.now()
            if not last_price:
                last_price = buy_price

            trade_data = TradeData(
                ticker=ticker,
                buy_date=buy_date,
                buy_price=buy_price,
                sell_date=last_date,
                sell_price=last_price,
                quantity=quantity,
                log_msg="open",
            )
            open_trades.append(trade_result(trade_data, self.fees, opened=True))
            total += last_price * quantity

        open_trades.sort(key=lambda x: x.buy_date)
        return {"positions_total": total, "open_trades": open_trades}

    def current_value(self):
        """Calculate current total portfolio value."""
        total = self.cash
        if self.positions:
            positions_value = self.get_open_positions()["positions_total"]
            total += positions_value
            if self.warn:
                print(
                    f"""Cash: {self.cash:.2f}, Positions: {positions_value:.2f}, Total: {
                        total:.2f
                    }"""
                )
        return total

    def get_tradelist(self):
        """Returns DataFrame of all trades."""
        closed_trades = [trade_result(trade, self.fees) for trade in self.trades]
        open_trades = self.get_open_positions()["open_trades"]

        all_trades = closed_trades + open_trades
        df = pd.DataFrame(all_trades, columns=TradeResult._fields)
        df = df.sort_values("sell_date").reset_index(drop=True)
        return df

    def has_position(self, ticker):
        """Check if ticker is in current positions."""
        return ticker in self.positions

    def __repr__(self):
        """String representation of portfolio."""
        print("---- PORTFOLIO ----")
        print(f"Total Value: {self.current_value():.2f}")
        print(f"Cash: {self.cash:.2f}")
        print(f"Positions Value: {self.get_open_positions()['positions_total']:.2f}")
        print(f"Total fees Paid: {self.total_fees_paid:.2f}")
        print(
            f"Return: {((self.current_value() / self.starting_capital - 1) * 100):.2f}%"
        )
        pprint(self.positions)
        return f"Last Date: {self.max_date}"

    def equity_line(self):
        """Generate equity line data."""
        dates, capital = [], []
        for date_obj, data in sorted(self.pos_history.items()):
            cash, positions = data
            total = cash + sum(price * qty for _, qty, price in positions)
            dates.append(date_obj)
            capital.append(total)
        return dates, capital

    def verify_consistency(self):
        """Verify portfolio calculations are consistent."""
        if not self.pos_history:
            return True

        dates, capital = self.equity_line()
        if dates and capital:
            equity_line_total = capital[-1]
            current_value_total = self.current_value()

            difference = abs(equity_line_total - current_value_total)
            if difference > 0.01:
                print("WARNING: Calculation mismatch!")
                print(f"Equity line latest: {equity_line_total:.2f}")
                print(f"Current sum: {current_value_total:.2f}")
                print(f"Difference: {difference:.2f}")
                return False

        self.starting_capital
        current_total = self.current_value()

        net_trading_profit = sum(
            trade_result(trade, self.fees).net_profit for trade in self.trades
        )
        open_positions_profit = sum(
            (last_price - buy_price) * qty - (buy_price * qty * self.fees)
            for _, qty, buy_price, _, last_price in self.positions.values()
        )

        if self.warn:
            print(f"Capital verification:")
            print(f"  Starting: {self.starting_capital:.2f}")
            print(f"  Current total: {current_total:.2f}")
            print(f"  Closed trades profit: {net_trading_profit:.2f}")
            print(f"  Open positions profit: {open_positions_profit:.2f}")
            print(f"  Total fees: {self.total_fees_paid:.2f}")

        return True

    def get_equity(self, show=False):
        return gen_equity(self, show=show)

    def stats_trades(self):
        return analyze_trades(self.get_tradelist(), show=False)

    def full_report(self, kind="html", outfile='report.html', title="Portfolio report"):
        if kind == "excel":
            return excel_report(self, title=title)
        return html_report(self, outfile=outfile, title=title)

    def basic_report(self, show=True):
        return summary(self, show=show)

    def base_results(self):
        dates, capital_line = self.equity_line()
        drawdown = utils.get_drawdown(np.array(capital_line))
        drawdown_val = drawdown["max_dd"]
        pf_return = pct_diff(self.starting_capital, self.current_value())
        return pf_return, drawdown_val


def calculate_annual_growth_rate(portfolio):
    """Calculate the annualized growth rate percentage."""
    try:
        start_date = min(portfolio.pos_history.keys())
        end_date = portfolio.max_date

        if start_date >= end_date:
            raise ValueError("Start date must be before end date")

        difference = relativedelta(end_date, start_date)
        months = difference.months + (difference.years * 12)
        total_years = difference.years + (months / 12.0)

        if total_years <= 0:
            months = max(months, 1)
            total_years = months / 12.0

        start_value = portfolio.starting_capital
        end_value = portfolio.current_value()

        if end_value < 0:
            raise ValueError("Ending value cannot be negative")

        if end_value == start_value:
            return 0.0

        growth_factor = end_value / start_value
        annual_growth_rate = (growth_factor ** (1 / total_years) - 1) * 100

        return annual_growth_rate

    except (AttributeError, TypeError) as e:
        raise ValueError("Invalid portfolio object structure") from e
    except Exception as e:
        raise ValueError(f"Error calculating growth rate: {str(e)}") from e


def summary(portfolio, show=False):
    """Generate portfolio summary statistics."""
    if not portfolio.pos_history or not portfolio.trades:
        print("!No position history available or no trades")
        return None

    portfolio.verify_consistency()

    start_date = min(portfolio.pos_history.keys())
    end_date = portfolio.max_date
    difference = relativedelta(end_date, start_date)
    months = difference.months + (difference.years * 12)
    years = max(difference.years + (months / 12.0), 1 / 12.0)

    end_equity = portfolio.current_value()

    net_profit = end_equity - portfolio.starting_capital
    net_profit_prc = pct_diff(portfolio.starting_capital, end_equity)
    all_trades = portfolio.get_tradelist()
    trades_num = len(all_trades)

    if trades_num == 0:
        if show:
            print("No trades executed")
        return None

    profitable = all_trades[all_trades["net_profit"] > 0]
    prc_profitable = len(profitable) / float(trades_num) * 100

    dates, capital_line = portfolio.equity_line()

    drawdown = utils.get_drawdown(np.array(capital_line))
    drawdown_val = drawdown["max_dd"]
    drawdown_start = dates[drawdown["start"]]

    summary = [
        [
            "Date Range:",
            f"{utils.datetime_to_str(start_date)} to {utils.datetime_to_str(end_date)}",
        ],
        ["Total Return (%):", f"{net_profit_prc:+.2f}%"],
        ["Ann. Ret. (%):", f"{calculate_annual_growth_rate(portfolio):.2f}%"],
        ["Max Drawdown:", f"{drawdown_val:.1f}%"],
        ["Winning Ratio (%):", f"{prc_profitable:.2f}%"],
        ["Max. Drawdown Start:", f"{utils.datetime_to_str(drawdown_start)}"],
        ["Net Profit:", f"{net_profit:,.2f}"],
        ["Total fees Paid:", f"{portfolio.total_fees_paid:,.2f}"],
        ["Starting Capital:", f"{portfolio.starting_capital:,.2f}"],
        ["Ending Capital:", f"{end_equity:,.2f}"],
        ["Number of Trades:", f"{trades_num:,}"],
        ["Fees Rate (%):", f"{portfolio.fees * 100:.2f}%"],
        ["Single Mode Enabled:", f"{portfolio.single}"],
    ]

    if show:
        pprint(summary)

    return summary


def gen_equity(portfolio, show=False):
    """Generate equity line data."""

    def calculate_eqt_val(portfolio):
        outlist = []
        prev_capital = portfolio.starting_capital

        for date_obj in sorted(portfolio.pos_history.keys()):
            cash, positions = portfolio.pos_history[date_obj]
            positions_amount = sum(price * qty for _, qty, price in positions)
            capital = cash + positions_amount

            if prev_capital > 0 and abs(capital - prev_capital) > prev_capital * 0.5:
                print(
                    f"!Warning: Large capital jump on {date_obj}: {prev_capital} -> {capital}")

            outlist.append(
                (
                    date_obj,
                    capital,
                    cash,
                    ", ".join(
                        f"({ticker}={price:.1f}*{qty:.1f})"
                        for ticker, qty, price in positions
                    ),
                )
            )
            prev_capital = capital

        return outlist

    events_history_diff = set(portfolio.events).difference(portfolio.pos_history)
    if events_history_diff:
        print(f"Warning: {len(events_history_diff)} events not in position history")

    equity = calculate_eqt_val(portfolio)
    if not equity:
        print("No equity data available")
        return pd.DataFrame()

    mdf = pd.DataFrame(equity, columns=["date_obj", "capital", "cash", "positions"])
    mdf["events"] = tuple([portfolio.events.get(x) for x in mdf.date_obj.values])
    mdf["pydate"] = mdf.date_obj.map(lambda x: pd.to_datetime(x))

    all_days = pd.date_range(mdf.pydate.min(), mdf.pydate.max(), freq="D")
    mdf.set_index("pydate", inplace=True)
    ndf = mdf.reindex(all_days)
    ndf["events"] = ndf["events"].fillna("-")
    ndf["date_obj"] = ndf["date_obj"].map(lambda x: utils.datetime_to_str(x))
    ndf.ffill(inplace=True)
    return ndf


def analyze_trades(trades, show=True):
    """Analyze trade statistics."""

    def remove_outliers(data, threshold=2.0):
        devs = np.abs(data - np.median(data))
        med_dev = np.median(devs)
        scaled = devs / med_dev if med_dev else 0.0
        return data[scaled < threshold]

    def consecutive_counter(direction="up"):
        count = 0
        mult = 1 if direction == "up" else -1

        def counter(value):
            nonlocal count
            if value * mult > 0:
                result = count
            else:
                count += 1
                result = None
            return result

        return counter

    def get_streaks(df, col):
        def summarize_group(group):
            return pd.Series(
                {
                    "from": str(group.index.min())[:10],
                    "to": str(group.index.max())[:10],
                    "count": group.index.size,
                }
            )

        profit_counter = consecutive_counter("up")
        loss_counter = consecutive_counter("down")

        df["cons_profits"] = df[col].map(profit_counter)
        df["cons_losses"] = df[col].map(loss_counter)

        try:
            top_profit = (
                df.groupby("cons_profits")
                .apply(summarize_group, include_groups=False)
                .nlargest(1, "count")
                .iloc[0]
                .tolist()
            )

            top_loss = (
                df.groupby("cons_losses")
                .apply(summarize_group, include_groups=False)
                .nlargest(1, "count")
                .iloc[0]
                .tolist()
            )
            top_profit[-1] = int(top_profit[-1])
            top_loss[-1] = int(top_loss[-1])
            return [
                f"Longest profit streak: {top_profit[-1]}",
                f"Longest loss streak: {top_loss[-1]}",
            ]
        except (IndexError, KeyError):
            return ["Longest profit streak: No data", "Longest loss streak: No data"]

    df = trades.copy()

    if len(df) == 0:
        return ["No trades to analyze"]

    df["buy_date"] = pd.to_datetime(df["buy_date"])
    df["sell_date"] = pd.to_datetime(df["sell_date"])

    df["duration"] = (df["sell_date"] - df["buy_date"]).dt.total_seconds()
    avg_duration_seconds = df["duration"].mean()

    if not pd.isna(avg_duration_seconds):
        minutes, seconds = divmod(avg_duration_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)

        duration_parts = []
        if days >= 1:
            duration_parts.append(f"{int(days)}d")
        if hours >= 1:
            duration_parts.append(f"{int(hours)}h")
        if minutes >= 1:
            duration_parts.append(f"{int(minutes)}m")
        if seconds >= 1 and days < 1:
            duration_parts.append(f"{int(seconds)}s")

        avg_duration_str = " ".join(duration_parts) if duration_parts else "<1s"
    else:
        avg_duration_str = "N/A"

    profits = df[df["net_profit"] > 0.0]["net_profit"]
    losses = df[df["net_profit"] < 0.0]["net_profit"]

    results = [
        f"Avg. profit: {profits.mean():.1f}, Avg. loss: {losses.mean():.1f}",
        f"Avg. trade duration: {avg_duration_str}",
    ]

    df["date"] = df["buy_date"]
    df.set_index("date", inplace=True)

    df["wday"] = df.index.map(lambda x: x.strftime("%A"))

    try:
        results.extend(get_streaks(df, "net_profit"))
    except Exception as e:
        print(f"Error calculating streaks: {e}")

    daily = df.groupby(level=0).apply(
        lambda g: pd.Series(
            data=[len(g.index), g["net_profit"].sum()], index=["count", "profit"]
        ),
        include_groups=False,
    )

    all_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    trading_days = len(daily.index)
    total_days = len(all_days)

    results.extend(
        [
            f"Total days: {total_days}, Trading days: {trading_days} ({
                trading_days /
                total_days *
                100:.0f}%)",
            f"Max trades per day: {
                daily['count'].max():.1f}",
        ])

    full_daily = pd.DataFrame(daily.reindex(all_days, fill_value=0))
    full_daily["week"] = full_daily.index.map(
        lambda x: f"{x.isocalendar()[0]}_{x.isocalendar()[1]}"
    )

    weekly = full_daily.groupby("week")["count"].sum()
    results.append(f"Max trades per week: {weekly.max():.1f}")

    try:
        weekly_clean = remove_outliers(weekly)
        results.append(f"Avg trades per week: {weekly_clean.mean():.1f}")
    except (KeyError, IndexError):
        print("Warning: Could not calculate avg trades per week")

    full_daily["year"] = full_daily.index.map(lambda x: x.year)

    yearly_trades = full_daily.groupby("year")["count"].sum()
    yearly_profit = full_daily.groupby("year")["profit"].sum()

    results.extend(
        [
            f"Avg trades per year: {yearly_trades.mean():.1f}",
            f"Avg profit per year: {yearly_profit.mean():.1f}",
        ]
    )

    if show:
        pprint(results)

    return pd.DataFrame(results, columns=["Trading Statistics"])


def get_year_stats(portfolio):
    """Generate yearly statistics."""
    equity_data = portfolio.get_equity()
    cleaned_equity_data = equity_data.drop(["date_obj"], axis=1)

    trade_data = portfolio.get_tradelist()
    trade_data["year"] = trade_data.sell_date.apply(
        lambda x: str(x)[:4] if pd.notna(x) and len(str(x)) >= 4 else None
    )

    yearly_capital_returns = (
        cleaned_equity_data.capital.groupby(pd.Grouper(freq="YE"))
        .apply(lambda group: pct_diff(group.iloc[0], group.iloc[-1]))
        .round(2)
    )

    year_return_pairs = list(
        zip(
            [str(date_index)[:4] for date_index in yearly_capital_returns.index],
            yearly_capital_returns.values,
        )
    )

    yearly_trade_profits = trade_data.groupby("year").net_profit.sum().to_dict()

    combined_yearly_stats = []
    for year_str, capital_return in year_return_pairs:
        trade_profit = yearly_trade_profits.get(year_str, 0)
        combined_yearly_stats.append([year_str, capital_return, trade_profit])

    return pd.DataFrame(combined_yearly_stats, columns=["year", "return", "net_return"])


def html_report(portfolio, outfile="portfolio-report.html", title="Portfolio report"):
    import json
    """Generate HTML report of portfolio performance and trades."""
    try:
        import df2tables as df2t
    except ModuleNotFoundError:
        print("\nError: df2tables module not found")
        print('Install with: "pip install df2tables" to generate full HTML report')
        print("Showing basic report only:")
        portfolio.basic_report()
        return

    def to_js_timestamps(dt_index):
        """Convert DatetimeIndex to JavaScript timestamps."""
        return (dt_index.astype("int64") // 10**6).tolist()

    def open_file(filename):
        import subprocess

        if sys.platform.startswith("win"):
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])

    # Generate basic metrics
    metrics_df = pd.DataFrame(
        portfolio.basic_report(show=False), columns=["Metric", "Value"]
    )

    def format_spans(value_str):
        """Apply color formatting to positive/negative values."""
        if value_str.startswith("+"):
            return f'<span class="positive">{value_str}</span>'
        if value_str.startswith("-"):
            return f'<span class="negative">{value_str}</span>'
        return value_str

    # Apply color formatting
    metrics_df["Value"] = metrics_df["Value"].map(lambda v: format_spans(v))

    # CSS class for tables
    pure_table = "pure-table"

    # Generate trade stats table
    trade_stats = portfolio.stats_trades().to_html(
        classes=[pure_table + " f_right"], index=False, border=0
    )

    # Prepare trades data
    trades_df = portfolio.get_tradelist()
    trades_df["buy_date"] = trades_df["buy_date"].map(lambda d: utils.datetime_to_str(d))
    trades_df["sell_date"] = trades_df["sell_date"].map(lambda d: utils.datetime_to_str(d))

    # Prepare chart data
    equity_data = portfolio.get_equity().capital
    chart_x = to_js_timestamps(equity_data.index)
    chart_y = equity_data.values.tolist()

    # Generate trades table
    trades_table = df2t.render_inline(
        trades_df,
        title="Portfolio Trades",
        dropdown_select_threshold=4,
        load_column_control=True,
        num_html=["net_profit", "profit_pct", "gross_profit"],
    )

    # Template data
    data = {
        "basic_report": metrics_df.reset_index(drop=True)
        .to_html(
            index=False,
            border=0,
            classes=[pure_table],
            escape=False,
        )
        .replace("<table ", '<table style="float:left;margin-right:2rem;" '),
        "trades": trades_table,
        "trades_stats": trade_stats,
        "real_dates": json.dumps(chart_x) ,
        "real_values": json.dumps(chart_y),
        "report_title": title,
        "generate_jsdata": "// generate_jsdata function",
    }

    template_file = "antreport_templ.html"
    try:
        # Python 3.9+
        from importlib import resources

        template_path = str(resources.files("antback").joinpath(template_file))
    except ImportError:
        # Fallback for older versions
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), template_file
        )

    write_from_template(template_path, outfile, data)
    open_file(outfile)


def excel_report(portfolio, outfile="portfolio-report.xlsx", title="Porfolio report"):
    """Generate Excel report of portfolio."""

    try:
        import xlreport as xl
    except ModuleNotFoundError:
        print("\nError: xlreport module not found")
        print('Install with: "pip install xlreport" to generate Excel report')
        print("Showing basic report only:")
        portfolio.basic_report()
        return
    excel_file = xl.Exfile(outfile)
    excel_file.write(
        portfolio.basic_report(show=False), title=title, worksheet_name="Basic"
    )
    excel_file.write(
        get_year_stats(portfolio).reset_index(drop=True),
        title="Yearly Performance",
        worksheet_name="Years",
    )

    trades_df = portfolio.get_tradelist()
    trades_df["buy_date"] = trades_df["buy_date"].map(lambda d: utils.datetime_to_str(d))
    trades_df["sell_date"] = trades_df["sell_date"].map(lambda d: utils.datetime_to_str(d))
    excel_file.write(trades_df, title="Trade List", worksheet_name="Trades")

    excel_file.write(
        portfolio.get_equity(), title="Equity Curve", worksheet_name="Equity"
    )

    excel_file.write(
        portfolio.stats_trades(), title="Trade Statistics", worksheet_name="Stats"
    )

    excel_file.add_links()
    excel_file.save(start=True)


def generate_test_data() -> Portfolio:
    import random
    from datetime import time
    from random import randint

    """Generate test data with guaranteed chronological order"""
    testp = Portfolio(100_000, single=False, warn=0)
    stocks = ["AAPL", "GOOG", "TSLA", "AMZN", "MSFT"]
    prices = {ticker: 100 * (0.8 + random.random() * 0.4) for ticker in stocks}
    trade_count = 0
    yearly_trade_targets = {2020: 15, 2021: 15, 2022: 15, 2023: 10}

    # Track last operation time to ensure chronological order
    last_op_time = datetime(2019, 12, 31)  # Initial date before our range

    def get_next_market_time(dt):
        """Get next valid market time after last_op_time"""
        # Market hours: 9:30-16:00
        base_time = dt.replace(hour=9, minute=30, second=0)

        # If requested time is before last op, move to next day
        if dt.date() < last_op_time.date():
            base_time = last_op_time + timedelta(days=1)
            base_time = base_time.replace(hour=9, minute=30, second=0)
        elif dt.date() == last_op_time.date():
            # Same day - ensure time is after last_op_time
            if base_time <= last_op_time:
                base_time = last_op_time + timedelta(minutes=1)

        # Ensure time is within market hours
        if base_time.time() < time(9, 30):
            base_time = base_time.replace(hour=9, minute=30)
        elif base_time.time() > time(16, 0):
            base_time = base_time.replace(hour=16, minute=0)

        return base_time

    # 1. Initial positions - properly sequenced
    fixed_trade = testp.cash / len(stocks)
    start_date = get_next_market_time(datetime(2020, 1, 1))

    for ticker in stocks:
        trade_time = get_next_market_time(start_date)
        testp.process("buy", ticker, trade_time, prices[ticker], buy_fixed=fixed_trade)
        trade_count += 1
        yearly_trade_targets[2020] -= 1
        last_op_time = trade_time

        # Update price after each buy
        update_time = last_op_time + timedelta(seconds=1)
        testp.update(ticker, update_time, prices[ticker])
        last_op_time = update_time

    # 2. Daily simulation
    current_date = start_date.date() + timedelta(days=1)
    while current_date <= datetime(2022, 12, 31).date():
        if current_date.weekday() < 5:  # Weekdays only
            year = current_date.year
            daily_ops = []

            # Market open update
            open_time = get_next_market_time(
                datetime.combine(current_date, time(9, 30))
            )
            for ticker in stocks:
                prices[ticker] = max(
                    1.02, prices[ticker] * (0.99 + random.random() * 0.0205)
                )
                daily_ops.append((open_time, "update", ticker))

            # Potential trades
            for ticker in stocks:
                trade_chance = 0.15 if yearly_trade_targets[year] > 0 else 0.05
                if random.random() < trade_chance and trade_count < 100:
                    action = "sell" if ticker in testp.positions else "buy"
                    trade_time = get_next_market_time(
                        datetime.combine(
                            current_date, time(randint(10, 15), randint(0, 59))
                        )
                    )
                    daily_ops.append((trade_time, action, ticker))

            # Market close update
            close_time = get_next_market_time(
                datetime.combine(current_date, time(16, 0))
            )
            for ticker in stocks:
                daily_ops.append((close_time, "update", ticker))

            # Execute in order with time validation
            for op_time, op_type, ticker in sorted(daily_ops, key=lambda x: x[0]):
                if op_time <= last_op_time:
                    op_time = last_op_time + timedelta(seconds=1)

                if op_type == "update":
                    testp.update(ticker, op_time, prices[ticker])
                elif op_type == "buy" and testp.cash > 1000:
                    testp.process(
                        "buy",
                        ticker,
                        op_time,
                        prices[ticker],
                        buy_fixed=random.uniform(testp.cash * 0.1, testp.cash * 0.5),
                        log_msg=f"{year} Trade {trade_count}",
                    )
                    trade_count += 1
                    yearly_trade_targets[year] -= 1
                elif op_type == "sell":
                    testp.process(
                        "sell",
                        ticker,
                        op_time,
                        prices[ticker],
                        log_msg=f"{year} Trade {trade_count}",
                    )
                    trade_count += 1
                    yearly_trade_targets[year] -= 1

                last_op_time = op_time

        current_date += timedelta(days=1)

    print(f"Generated {trade_count} trades")
    print(f"Last operation time: {last_op_time}")
    return testp

def demo():
    test_pf  = generate_test_data()
    print(test_pf.basic_report())
    test_pf.full_report("html")
    # test_pf.full_report('excel')

if __name__ == "__main__":
    demo()
