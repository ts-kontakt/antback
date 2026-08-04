#!/usr/bin/python
# coding=utf-8
from collections import defaultdict, namedtuple
from datetime import date, datetime, timedelta
from math import isnan
from pprint import pprint

import numpy as np
import pandas as pd

try:
    from . import utils
except ImportError:
    import utils

TradeData = namedtuple(
    "TradeData",
    ["ticker", "open_date", "buy_price", "close_date", "sell_price", "quantity", "log_msg", ],
)

TradeResult = namedtuple("TradeResult", [
    "ticker", "open_date", "close_date", "buy_price", "sell_price", "quantity", "position_size",
    "net_profit", "profit_pct", "gross_profit", "buy_fees", "sell_fees", "total_fees", "log_msg",
],
                         )

# Reason codes returned by buy()/sell() on rejection.
ALREADY_HELD = "already_held"
ZERO_QUANTITY = "zero_quantity"
DUST_QUANTITY = "dust_quantity"
INSUFFICIENT_CASH = "insufficient_cash"
NOT_HELD = "not_held"

RejectedOrder = namedtuple("RejectedOrder", ["ticker", "date", "action", "reason"])

def trade_result(trade, brokerage, opened=False):
    """Calculate trade results including commissions."""
    position_size = trade.buy_price * trade.quantity
    sell_value = trade.sell_price * trade.quantity
    buy_fees = position_size * brokerage

    if not opened:
        sell_fees = sell_value * brokerage
        total_fees = buy_fees + sell_fees
    else:
        sell_fees = 0
        total_fees = buy_fees

    gross_profit = sell_value - position_size
    net_profit = gross_profit - total_fees

    # ── Net return on total capital deployed (includes buy-side commission) ──
    cost_basis = position_size + buy_fees
    profit_pct = (net_profit / cost_basis) * 100 if cost_basis != 0 else float("nan")

    log_msg = "*opened" if opened else trade.log_msg

    return TradeResult(
        ticker=trade.ticker,
        open_date=trade.open_date,
        close_date=trade.close_date,
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
    __slots__ = ("warn", "starting_capital", "cash", "positions", "fees", "max_date", "trades",
                 "pos_history", "single", "events", "total_fees_paid", "allow_fractional",
                 "rejections", "not_held_count",
                 )
    """Trading portfolio management class."""
    DEFAULT_PERCENT = 1.0
    MIN_AMOUNT = 500

    def __init__(self, cash, single=True, warn=False, allow_fractional=False, fees=0.0015):
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
        self.rejections = []
        self.not_held_count = 0

    def _validate_and_update_date(self, date_obj):
        """Validate date format and update max_date."""
        # date_obj = self._normalize_date(date_obj)
        if not isinstance(date_obj, (date, datetime)):
            raise ValueError("'date' or 'datetime' allowed. Convert using pd.to_datetime()")

        if self.max_date and date_obj < self.max_date:
            print(f"! Received date: {date_obj}, self.max_date: {self.max_date}")
            raise ValueError("Date not in chronological order")

        if self.max_date is None or date_obj > self.max_date:
            self.max_date = date_obj

        return date_obj

    def buy(self, ticker, date_obj, price, fixed_val=None):
        """Buy shares of a ticker."""
        if fixed_val is None:
            if not self.single:
                raise AssertionError("For multi ticker portfolio fixed_val buy value is expected")

        if price <= 0 or isnan(price):
            raise ValueError("Price must be positive")

        date_obj = self._validate_and_update_date(date_obj)

        if self.single and self.positions and ticker not in self.positions:
            raise Exception(f"Single mode - can't add new ticker: {ticker}")

        if ticker in self.positions:
            if self.warn:
                print(f"! Re-buying not allowed: {ticker} {date_obj}")
            return ALREADY_HELD

        if fixed_val is not None:
            # assert price < fixed_val, f"fixed_val {fixed_val}  must be >= price {price} "
            if price > fixed_val:
                print(f"!warning: price {price}  >= fixed_val {fixed_val}")
            assert fixed_val >= self.MIN_AMOUNT, f"""fixed_val amount must be >= {
                self.MIN_AMOUNT
            }"""
            if self.cash < fixed_val:
                msg = f"{date_obj} - Insufficient cash: {self.cash:.1f} < {fixed_val}"
                self.events[date_obj].append(msg)
                self.rejections.append(RejectedOrder(ticker, date_obj, "buy", INSUFFICIENT_CASH))
                raise ValueError(msg)
            available_investment = fixed_val
        else:
            available_investment = self.cash * self.DEFAULT_PERCENT  # 1.0

        buffer_factor = 0.99999  # Prevent error comparing floats
        max_affordable_shares = (available_investment * buffer_factor) / (price * (1 + self.fees))

        quantity = (max_affordable_shares if self.allow_fractional else int(max_affordable_shares))

        if quantity == 0 and not self.allow_fractional:
            msg = f"! 0 quantity, cash: {self.cash}, can set: self.allow_fractional=True"
            self.events[date_obj].append(msg)
            self.rejections.append(RejectedOrder(ticker, date_obj, "buy", ZERO_QUANTITY))
            if self.warn:
                print(msg)
            return ZERO_QUANTITY
        elif quantity <= 0.001:
            msg = f"""! buy quantity <= 0.001 for {ticker} at {price}: need {
                price * (1 + self.fees)
            },
             have {available_investment:.2f}"""
            self.events[date_obj].append(msg)
            self.rejections.append(RejectedOrder(ticker, date_obj, "buy", DUST_QUANTITY))
            if self.warn:
                print(msg)
            return DUST_QUANTITY

        share_cost = price * quantity
        fees = share_cost * self.fees
        total_cost = share_cost + fees

        if self.cash < total_cost:
            msg = f"Calculation error: cash {self.cash:.5f} < needed {total_cost:.5f}"
            self.events[date_obj].append(msg)
            self.rejections.append(RejectedOrder(ticker, date_obj, "buy", INSUFFICIENT_CASH))
            raise ValueError(msg)

        self.cash -= total_cost
        self.total_fees_paid += fees
        self.positions[ticker] = [date_obj, quantity, price, date_obj, price]

        self.events[date_obj].append(
            f"buy: {ticker}({price:.2f}*{quantity:.1f}, cost: {total_cost:.1f}, fees: {fees:.1f})")

        if self.warn:
            print(f"""Bought {quantity:.4f} shares of {ticker} at {price:.2f},
                total cost: {total_cost:.2f}""")

        self.save_positions(date_obj)

    def sell(self, ticker, date_obj, price, log_msg=""):
        """Sell shares of a ticker."""
        date_obj = self._validate_and_update_date(date_obj)
        if price <= 0 or isnan(price):
            raise ValueError("Price must be positive")

        if ticker not in self.positions:
            self.not_held_count += 1
            if self.warn:
                print(f"! Sell rejected, not held: {ticker} {date_obj}")
            return NOT_HELD

        open_date, quantity, buy_price, last_date, last_price = self.positions.pop(ticker)

        gross_proceeds = price * quantity
        fees = gross_proceeds * self.fees
        net_proceeds = gross_proceeds - fees

        self.trades.append(
            TradeData(ticker, open_date, buy_price, date_obj, price, quantity, log_msg))

        self.cash += net_proceeds
        self.total_fees_paid += fees

        self.events[date_obj].append(
            f"sell: {ticker}({price:.2f}*{quantity:.1f}, proceeds: {net_proceeds:.1f}, fees: {fees:.1f})"
        )

        if self.warn:
            print(f"""Sold {quantity:.4f} shares of {ticker} at {price:.2f},
                 net proceeds: {net_proceeds:.2f}""")

        self.save_positions(date_obj)

    def update(self, ticker, date_obj, price):
        """Update price for existing position."""
        date_obj = self._validate_and_update_date(date_obj)
        if price <= 0 or isnan(price):
            raise ValueError("Price must be positive")

        changed = False
        pos = self.positions.get(ticker)

        if pos is not None:
            pos[-2] = date_obj  # Always update last_date
            if pos[-1] != price:
                pos[-1] = price
                changed = True

        # Save if it's the first time we've seen this timestamp (heartbeat)
        # OR if a held ticker's price actually mutated
        if changed or date_obj not in self.pos_history:
            self.save_positions(date_obj)

    def process(self, signal, ticker, date_obj, price, buy_fixed=None, log_msg=""):
        """Unified entry-point for any daily signal."""
        if signal == "update" or signal is None:
            return self.update(ticker, date_obj, price)
        elif signal == "buy":
            return self.buy(ticker, date_obj, price, fixed_val=buy_fixed)
        elif signal == "sell":
            return self.sell(ticker, date_obj, price, log_msg=log_msg)
        else:
            raise ValueError(f"Unknown signal {signal!r}")

    def save_positions(self, record_date):
        """Save current portfolio state."""
        pos_list = []
        for ticker, (open_date, qty, buy_price, last_date, last_price, ) in self.positions.items():
            pos_list.append((ticker, qty, last_price))
        self.pos_history[record_date] = (self.cash, pos_list)

    def get_open_positions(self):
        """Get information about open positions."""
        if not self.positions:
            return {"positions_total": 0, "open_trades": []}

        open_trades = []
        total = 0

        for ticker, (open_date, quantity, buy_price, last_date, last_price,
                     ) in self.positions.items():
            if not last_date:
                last_date = self.max_date or datetime.now()
            if not last_price:
                last_price = buy_price

            trade_data = TradeData(ticker=ticker,
                                   open_date=open_date,
                                   buy_price=buy_price,
                                   close_date=last_date,
                                   sell_price=last_price,
                                   quantity=quantity,
                                   log_msg="open",
                                   )
            open_trades.append(trade_result(trade_data, self.fees, opened=True))
            total += last_price * quantity

        open_trades.sort(key=lambda x: x.open_date)
        return {"positions_total": total, "open_trades": open_trades}

    def current_value(self):
        """Calculate current total portfolio value."""
        total = self.cash
        if self.positions:
            positions_value = self.get_open_positions()["positions_total"]
            total += positions_value
            if self.warn:
                print(f"""Cash: {self.cash:.2f}, Positions: {positions_value:.2f},
                 Total: {total:.2f}""")
        return total

    def get_tradelist(self):
        """Returns DataFrame of all trades."""
        closed_trades = [trade_result(trade, self.fees) for trade in self.trades]
        open_trades = self.get_open_positions()["open_trades"]

        all_trades = closed_trades + open_trades
        df = pd.DataFrame(all_trades, columns=TradeResult._fields)
        df = df.sort_values("close_date").reset_index(drop=True)
        return df

    def has_position(self, ticker):
        """Check if ticker is in current positions."""
        return ticker in self.positions

    def __repr__(self):
        """Concise, unambiguous representation (e.g. for debugging containers)."""
        return (
            f"Portfolio(cash={self.cash:,.2f}, "
            f"positions={len(self.positions)}, "
            f"max_date={self.max_date})"
        )

    def __str__(self):
        """Human-readable summary, printed by str() and print()."""
        current_val = self.current_value()
        positions_val = current_val - self.cash
        ret_pct = (current_val / self.starting_capital - 1) * 100

        lines = [
            "---- PORTFOLIO ----",
            f"Total Value:       {current_val:>12,.2f}",
            f"Cash:              {self.cash:>12,.2f}",
            f"Positions Value:   {positions_val:>12,.2f}",
            f"Total Fees Paid:   {self.total_fees_paid:>12,.2f}",
            f"Return:            {ret_pct:>11.2f}%",
            f"Last Date:         {self.max_date}",
            f"Positions:         {list(self.positions.keys())}",
        ]
        return "\n".join(lines)


    def equity_line(self):
        """Generate equity line data."""
        # pos_history keys are inserted in chronological order already --
        # doesn't move it), so no sort is needed here.
        dates, capital = [], []
        for date_obj, data in self.pos_history.items():
            cash, positions = data
            total = cash + sum(price * qty for _, qty, price in positions)
            dates.append(date_obj)
            capital.append(total)
        return dates, capital

    def get_equity(self, show=False):
        return gen_equity(self, show=show)

    def stats_trades(self):
        return utils.analyze_trades(self.get_tradelist(), show=False)

    def full_report(self, kind="html", outfile="report.html", title="Portfolio report"):
        if kind == "excel":
            return utils.excel_report(self, title=title)
        return utils.html_report(self, outfile=outfile, title=title)

    def basic_report(self, show=True):
        return utils.summary(self, show=show)

    def base_results(self):
        dates, capital_line = self.equity_line()
        drawdown = utils.get_drawdown(np.array(capital_line))
        drawdown_val = drawdown["max_dd"]
        pf_return = utils.pct_diff(self.starting_capital, self.current_value())
        return pf_return, drawdown_val


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
                print(f"!Warning: Large capital jump on {date_obj}: {prev_capital} -> {capital}")

            outlist.append(
                (date_obj, capital, cash, ", ".join(f"({ticker}={price:.1f}*{qty:.1f})"
                                                    for ticker, qty, price in positions),
                 ))
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

    # Add events information leveraging Pandas native vectorization
    events_series = pd.Series({k: str(v) for k, v in portfolio.events.items()}, dtype=object)
    mdf["events"] = mdf["date_obj"].map(events_series).fillna("")

    mdf["pydate"] = pd.to_datetime(mdf["date_obj"]).dt.date
    gdf = mdf.groupby(["pydate"]).agg({
        "date_obj": "last",
        "capital": "last",
        "cash": "last",
        "positions": "last",
        "events": lambda x: "".join(x),
    })

    all_days = pd.date_range(mdf.pydate.min(), mdf.pydate.max(), freq="D")
    ndf = gdf.reindex(all_days).ffill()
    ndf["date_obj"] = ndf["date_obj"].map(utils.datetime_to_str)
    return ndf


def generate_test_data():
    import random
    from datetime import time
    from random import randint
    """Generate test data with guaranteed chronological order"""
    testp = Portfolio(100_000, single=False, warn=1)
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
            open_time = get_next_market_time(datetime.combine(current_date, time(9, 30)))
            for ticker in stocks:
                prices[ticker] = max(1.02, prices[ticker] * (0.99 + random.random() * 0.0205))
                daily_ops.append((open_time, "update", ticker))

            # Potential trades
            for ticker in stocks:
                trade_chance = 0.15 if yearly_trade_targets[year] > 0 else 0.05
                if random.random() < trade_chance and trade_count < 100:
                    action = "sell" if ticker in testp.positions else "buy"
                    trade_time = get_next_market_time(
                        datetime.combine(current_date, time(randint(10, 15), randint(0, 59))))
                    daily_ops.append((trade_time, action, ticker))

            # Market close update
            close_time = get_next_market_time(datetime.combine(current_date, time(16, 0)))
            for ticker in stocks:
                daily_ops.append((close_time, "update", ticker))

            # Execute in order with time validation
            for op_time, op_type, ticker in sorted(daily_ops, key=lambda x: x[0]):
                if op_time <= last_op_time:
                    op_time = last_op_time + timedelta(seconds=1)

                if op_type == "update":
                    testp.update(ticker, op_time, prices[ticker])
                elif op_type == "buy" and testp.cash > 1000:
                    testp.process("buy",
                                  ticker,
                                  op_time,
                                  prices[ticker],
                                  buy_fixed=random.uniform(testp.cash * 0.1, testp.cash * 0.5),
                                  log_msg=f"{year} Trade {trade_count}",
                                  )
                    trade_count += 1
                    yearly_trade_targets[year] -= 1
                elif op_type == "sell":
                    testp.process("sell",
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
    test_pf = generate_test_data()
    print(test_pf.basic_report())
    test_pf.full_report("html")
    test_pf.full_report("excel")


def test_mixed_date_types_throws_error():
    pf = Portfolio(100000)
    pf.buy("AAPL", datetime(2020, 1, 1), 100, fixed_val=10000)
    try:
        pf.sell("AAPL", date(2020, 1, 5), 110)
    except TypeError as e:
        print("Caught expected TypeError:", e)
    else:
        raise AssertionError("Expected TypeError not raised")


if __name__ == "__main__":
    test_mixed_date_types_throws_error()
    demo()